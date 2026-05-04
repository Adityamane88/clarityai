import { ExternalLink, ImageOff, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ImageResult } from '../types'

interface ImageGalleryProps {
  images: ImageResult[]
  // Optional caption shown above the gallery (e.g. "Images of golden retrievers").
  caption?: string
}

/**
 * Compact image gallery used inside assistant message bubbles.
 *
 * Features:
 * - Lazy-loaded images.
 * - Per-image error fallback (broken hotlinks happen all the time).
 * - Click to open a lightbox; the lightbox links out to the source page.
 * - Keyboard support: Escape closes the lightbox.
 */
export default function ImageGallery({ images, caption }: ImageGalleryProps) {
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null)
  const [failed, setFailed] = useState<Record<number, boolean>>({})

  useEffect(() => {
    if (lightboxIndex === null) return undefined
    if (typeof window === 'undefined') return undefined

    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setLightboxIndex(null)
      if (event.key === 'ArrowRight') {
        setLightboxIndex((current) => {
          if (current === null) return current
          return Math.min(current + 1, images.length - 1)
        })
      }
      if (event.key === 'ArrowLeft') {
        setLightboxIndex((current) => {
          if (current === null) return current
          return Math.max(current - 1, 0)
        })
      }
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightboxIndex, images.length])

  if (!images || images.length === 0) return null

  // Show all images; failures degrade to a small placeholder card.
  const visible = images

  return (
    <div className="image-gallery" role="region" aria-label="Image results">
      {caption ? <div className="image-gallery-caption">{caption}</div> : null}
      <div className="image-gallery-grid">
        {visible.map((image, index) => {
          const broken = failed[index]
          return (
            <button
              key={`${image.url}-${index}`}
              className="image-card"
              type="button"
              onClick={() => setLightboxIndex(index)}
              title={image.title || image.domain || 'Image source'}
              aria-label={`Open image ${index + 1} from ${image.domain || 'source'}`}
            >
              {broken ? (
                <div className="image-card-broken" aria-hidden="true">
                  <ImageOff size={20} />
                  <span>{image.domain || 'unavailable'}</span>
                </div>
              ) : (
                <img
                  src={image.thumbnail || image.url}
                  alt={image.title || 'Search result image'}
                  loading="lazy"
                  decoding="async"
                  referrerPolicy="no-referrer"
                  onError={() => setFailed((prev) => ({ ...prev, [index]: true }))}
                />
              )}
              <span className="image-card-meta">
                {image.domain || 'source'}
              </span>
            </button>
          )
        })}
      </div>

      {lightboxIndex !== null ? (
        <div
          className="image-lightbox"
          role="dialog"
          aria-modal="true"
          aria-label="Image viewer"
          onClick={() => setLightboxIndex(null)}
        >
          <div className="image-lightbox-frame" onClick={(e) => e.stopPropagation()}>
            <button
              className="image-lightbox-close"
              type="button"
              onClick={() => setLightboxIndex(null)}
              aria-label="Close image"
            >
              <X size={18} />
            </button>
            <img
              className="image-lightbox-img"
              src={visible[lightboxIndex].url}
              alt={visible[lightboxIndex].title || 'Search result image'}
              referrerPolicy="no-referrer"
              onError={() => setFailed((prev) => ({ ...prev, [lightboxIndex]: true }))}
            />
            <div className="image-lightbox-caption">
              <div className="image-lightbox-title">
                {visible[lightboxIndex].title || 'Untitled'}
              </div>
              {visible[lightboxIndex].source_url ? (
                <a
                  href={visible[lightboxIndex].source_url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="image-lightbox-source"
                >
                  <ExternalLink size={13} /> Open source ({visible[lightboxIndex].domain || 'web'})
                </a>
              ) : null}
            </div>
            <div className="image-lightbox-pager">
              {lightboxIndex + 1} / {visible.length}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
