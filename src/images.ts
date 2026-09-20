import type { JsonValue } from "./types";
import { RequestValidationError } from "./types";

/**
 * A LocalJev extension, not part of the Jev protocol: an image travels inside
 * `state` in the OpenAI content-part shape, so the official TypeSafe SDKs — which
 * forward `state` as opaque JSON — can send one without any SDK change.
 */
export interface ImagePart {
  type: "image_url";
  image_url: { url: string };
}

export interface ExtractedImage {
  /**
   * Internally generated identifier, e.g. `image-1`. Never derived from the
   * request: the label introducing the image sits outside the `<document>`
   * wrapper, so anything caller-controlled in it would escape the
   * untrusted-data framing. Keys stay inside the document, where a placeholder
   * carrying this id marks the spot the image occupied.
   */
  id: string;
  part: ImagePart;
}

export interface ImageLimits {
  maxImages: number;
  maxImageBytes: number;
}

/** `data:image/png;base64,…` — the only form `fm serve` accepts. */
const DATA_URL_PREFIX = /^data:image\/(png|jpeg);base64,/;
const BASE64_ALPHABET = /^[A-Za-z0-9+/]*={0,2}$/;

/**
 * Depth and node ceilings for the `state` walk. A document nested tens of
 * thousands deep is a denial-of-service shape, not a real Jev state. 64 is far
 * past anything a hand-built or SDK-built state reaches, and 100_000 nodes
 * bounds the traversal of a body that is already limited in bytes.
 */
const MAX_DEPTH = 64;
const MAX_NODES = 100_000;

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * True for anything claiming to be an image part, however malformed. Detection
 * is deliberately looser than {@link isImagePart} so a near-miss is reported as
 * a broken image rather than silently serialized into the document text.
 */
function claimsToBeImage(value: unknown): value is Record<string, unknown> {
  return object(value) && value.type === "image_url";
}

/** True only for the exact accepted shape: `type` + `image_url.url`, nothing else. */
export function isImagePart(value: unknown): value is ImagePart {
  if (!claimsToBeImage(value)) return false;
  const keys = Object.keys(value);
  const hasOnlyKnownKeys =
    keys.length === 2 && keys.includes("type") && keys.includes("image_url");
  if (!hasOnlyKnownKeys) return false;
  const target = value.image_url;
  if (!object(target)) return false;
  const urlOnly = Object.keys(target).length === 1 && "url" in target;
  return urlOnly && typeof target.url === "string";
}

function locOf(path: (string | number)[]): (string | number)[] {
  // Kept as separate segments rather than joined into one string: `{"a.b": …}`
  // and `{"a": {"b": …}}` would otherwise report the identical location.
  return ["body", "state", ...path];
}

/**
 * Decoded byte count of a base64 payload, computed from its length so an
 * oversized image is rejected before anything allocates it.
 */
function decodedBytes(base64: string): number {
  const padding = base64.endsWith("==") ? 2 : base64.endsWith("=") ? 1 : 0;
  return Math.floor((base64.length * 3) / 4) - padding;
}

/**
 * Validates the base64 body by shape alone. Why not `atob`: it allocates the
 * whole image to prove what length, padding and alphabet already establish, and
 * every byte would then be decoded again on the way to the model.
 */
function isBase64(body: string): boolean {
  const hasLegalLength = body.length > 0 && body.length % 4 === 0;
  if (!hasLegalLength) return false;
  // The anchored pattern already allows "=" only as the last one or two
  // characters. Why not strip padding with /=+$/ first: unanchored at the front,
  // it retries from every "=" of a long run and turns quadratic (32k of "="
  // followed by "A" took half a second of synchronous work).
  return BASE64_ALPHABET.test(body);
}

function validateUrl(
  url: string,
  path: (string | number)[],
  limits: ImageLimits,
): void {
  const prefix = DATA_URL_PREFIX.exec(url);
  // Why not accept http(s) URLs: `fm serve` requires the bytes inlined, and
  // fetching a caller-supplied URL server-side would make LocalJev an SSRF hop.
  if (!prefix) {
    throw new RequestValidationError(
      locOf([...path, "image_url", "url"]),
      "Image URL must be an inline data URL of the form 'data:image/(png|jpeg);base64,<bytes>'",
    );
  }
  const body = url.slice(prefix[0].length);
  // Size first: it is O(1) from the length, so an oversized payload never
  // reaches the character scan below.
  const bytes = decodedBytes(body);
  if (bytes > limits.maxImageBytes) {
    throw new RequestValidationError(
      locOf([...path, "image_url", "url"]),
      `Image is ${bytes} bytes decoded, over the LOCALJEV_MAX_IMAGE_BYTES limit of ${limits.maxImageBytes}`,
    );
  }
  if (!isBase64(body)) {
    throw new RequestValidationError(
      locOf([...path, "image_url", "url"]),
      "Image data is not valid base64",
    );
  }
}

/**
 * Picks an id prefix that no string anywhere in `state` already contains, so a
 * caller cannot forge a reference to an image by writing `[image-1]` into their
 * own text. Deterministic: the same request must produce the same body, and so
 * the same seed.
 */
function chooseMarker(collides: (marker: string) => boolean): string {
  if (!collides("image")) return "image";
  for (let index = 0; index < 26; index += 1) {
    const candidate = `image_${String.fromCharCode(97 + index)}`;
    if (!collides(candidate)) return candidate;
  }
  // 27 distinct colliding prefixes in one state is adversarial, not accidental.
  throw new RequestValidationError(
    ["body", "state"],
    "Could not find an unused image placeholder prefix; remove text resembling '[image-N]' from state",
  );
}

interface Walk {
  images: ExtractedImage[];
  /** `state` with every image replaced by its `[<marker>-N]` placeholder. */
  document: JsonValue;
}

/** A container whose children are still to be visited. */
interface Frame {
  source: Record<string | number, JsonValue>;
  target: Record<string | number, JsonValue>;
  keys: (string | number)[];
  next: number;
  path: (string | number)[];
}

/** Where an image sat, so its placeholder can be written once the id is known. */
interface Slot {
  container: Record<string | number, JsonValue> | null;
  key: string | number;
}

/**
 * Splits `state` into the images to send as content parts and the document that
 * replaces each with an id placeholder. In normal mode nothing is extracted and
 * any image part is a 422 instead.
 *
 * The walk is iterative: recursion overflowed the stack on a deeply nested
 * document, and building a path string at every node made it O(depth²). A path
 * is assembled only where one is actually reported.
 */
export function extractImages(
  state: JsonValue,
  limits: ImageLimits,
  extended: boolean,
): Walk {
  const images: ExtractedImage[] = [];
  const slots: Slot[] = [];
  // Every string the caller supplied, keys included, so the marker can be
  // chosen to collide with none of them.
  const seen: string[] = [];
  let nodes = 0;
  let root: JsonValue = null;

  const visit = (
    value: JsonValue,
    path: (string | number)[],
    depth: number,
    container: Record<string | number, JsonValue> | null,
    key: string | number,
  ): Frame | null => {
    nodes += 1;
    if (nodes > MAX_NODES) {
      throw new RequestValidationError(
        locOf(path),
        `State holds more than the limit of ${MAX_NODES} values`,
      );
    }
    const put = (replacement: JsonValue): void => {
      if (container === null) root = replacement;
      else container[key] = replacement;
    };

    if (claimsToBeImage(value)) {
      if (!extended) {
        // Why not quietly treat it as text: the base64 would flow into the
        // document, blowing the context or producing a nonsense answer with no
        // hint of the cause.
        throw new RequestValidationError(
          locOf(path),
          "Image parts require LOCALJEV_EXTENSIONS=images",
        );
      }
      if (!isImagePart(value)) {
        throw new RequestValidationError(
          locOf(path),
          'An image part must be exactly {"type":"image_url","image_url":{"url":"data:image/(png|jpeg);base64,<bytes>"}}',
        );
      }
      if (images.length >= limits.maxImages) {
        throw new RequestValidationError(
          locOf(path),
          `Request carries more than the LOCALJEV_MAX_IMAGES limit of ${limits.maxImages} images`,
        );
      }
      validateUrl(value.image_url.url, path, limits);
      // The id depends only on discovery order, so it is stable for a given
      // request and carries nothing the caller wrote.
      images.push({ id: "", part: value });
      slots.push({ container, key });
      put(null);
      return null;
    }
    if (typeof value === "string") {
      seen.push(value);
      put(value);
      return null;
    }
    const isContainer = Array.isArray(value) || object(value);
    if (!isContainer) {
      put(value);
      return null;
    }
    if (depth >= MAX_DEPTH) {
      throw new RequestValidationError(
        locOf(path),
        `State is nested deeper than the limit of ${MAX_DEPTH}`,
      );
    }
    if (Array.isArray(value)) {
      const target: JsonValue[] = [];
      put(target);
      return {
        source: value as unknown as Record<string | number, JsonValue>,
        target: target as unknown as Record<string | number, JsonValue>,
        keys: value.map((_item, index) => index),
        next: 0,
        path,
      };
    }
    const keys = Object.keys(value);
    for (const name of keys) seen.push(name);
    // A null prototype so a `__proto__` key is written as an own data property
    // rather than silently redirected into the prototype and lost. JSON.stringify
    // treats it like any other object, so the document is unaffected otherwise.
    const target = Object.create(null) as Record<string, JsonValue>;
    put(target);
    return {
      source: value as Record<string | number, JsonValue>,
      target: target as Record<string | number, JsonValue>,
      keys,
      next: 0,
      path,
    };
  };

  const stack: Frame[] = [];
  const first = visit(state, [], 0, null, "");
  if (first) stack.push(first);

  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!;
    if (frame.next >= frame.keys.length) {
      stack.pop();
      continue;
    }
    const key = frame.keys[frame.next]!;
    frame.next += 1;
    const child = visit(
      frame.source[key] as JsonValue,
      [...frame.path, key],
      stack.length,
      frame.target,
      key,
    );
    if (child) stack.push(child);
  }

  // Why not return the rebuilt copy: an image-free request must serialize
  // exactly as it did before this extension existed, and the rebuild turns a
  // `__proto__` key into a prototype write that drops it from the document.
  if (images.length === 0) return { images, document: state };

  const marker = chooseMarker((candidate) =>
    seen.some((value) => value.includes(`[${candidate}-`)),
  );
  images.forEach((image, index) => {
    image.id = `${marker}-${index + 1}`;
    const slot = slots[index]!;
    const placeholder = `[${image.id}]`;
    if (slot.container === null) root = placeholder;
    else slot.container[slot.key] = placeholder;
  });
  return { images, document: root };
}
