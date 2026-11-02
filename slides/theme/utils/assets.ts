/**
 * Resolves asset URLs to work correctly with Vite's BASE_URL
 * in both development and production environments.
 *
 * @param url - The asset URL to resolve. Can be absolute (starting with /) or relative.
 * @returns The resolved URL with BASE_URL prepended for absolute paths
 *
 * @example
 * // In development (BASE_URL = '/')
 * resolveAssetUrl('/images/logo.png') // returns '/images/logo.png'
 *
 * // In production with base path (BASE_URL = '/my-slides/')
 * resolveAssetUrl('/images/logo.png') // returns '/my-slides/images/logo.png'
 *
 * // Relative URLs are returned as-is
 * resolveAssetUrl('./logo.png') // returns './logo.png'
 */
export function resolveAssetUrl(url?: string): string | undefined {
  if (!url) return url;
  if (url.startsWith("/")) {
    return import.meta.env.BASE_URL + url.slice(1);
  }
  return url;
}
