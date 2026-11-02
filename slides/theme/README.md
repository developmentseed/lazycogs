# Slidev Theme: DevelopmentSeed

A minimal Slidev theme for DevelopmentSeed presentations, featuring brand colors, typography, and custom layouts.

**Philosophy:**
This theme takes a minimal approach - it only customizes what's necessary for branding while leveraging Slidev's default layouts and features wherever possible. This keeps the codebase small and maintainable while ensuring compatibility with future Slidev updates.

**What this theme provides:**
- DevelopmentSeed brand colors and Roboto typography
- 2 custom layouts (`cover`, `title`)
- Reuses default layouts for everything else (`image-left`, `image-right`, `two-cols`, `center`, `default`, etc.)
- CSS enhancements and utility classes (`image-narrow`, brand color backgrounds)

## Features

### Brand Colors

- **Primary**: `#CF3F02` (Orange)
- **Secondary**: `#E2C044` (Yellow)
- **Danger**: `#A71D31` (Red)
- **Success**: `#4DA167` (Green)
- **Info**: `#2E86AB` (Blue)
- **Muted**: `#565B65` (Gray)

### Typography

- **Headings**: Roboto Condensed
- **Body**: Roboto
- **Code**: Roboto Mono

### Custom Layouts

#### `title`
A unique title layout with:
- Content in the top 2/3 of the slide
- Image in the bottom 1/3
- Support for subtitle in a two-column grid

```yaml
---
layout: title
image: /path/to/image.jpg
subtitle: Your subtitle here
---
```

#### `cover`
Enhanced cover layout with:
- Background image/gradient/color support
- **Automatic text color detection**:
  - Images → white text
  - Light colors → black text
  - Dark colors → white text
- Optional dim overlay for images
- Monaco editor text colors preserved
- Centered content

```yaml
---
layout: cover
background: /path/to/image.jpg
dim: true  # Optional: adds dark overlay to images for better contrast
---

# Or with a color background
---
layout: cover
background: '#E2C044'  # Automatically uses black text (light color)
---
```

### Layout Utilities

#### `image-narrow` class
Use with default `image-left` or `image-right` layouts to make the image take only 1/3 of the screen:

```yaml
---
layout: image-left
image: /path/to/image.jpg
class: image-narrow
---
```

### Components

- **LogoHorPos**: Horizontal positive logo
- **LogoHorNegMono**: Horizontal negative monochrome logo
- **LogoHorPosMono**: Horizontal positive monochrome logo
- **LogoHorDemi**: Horizontal demi logo
- **DecorativeRectangle**: Brand-colored decorative rectangles with customizable size, position, and blend modes

## Usage

In your `slides.md`:

```yaml
---
theme: ./theme
---
```

The theme extends Slidev's default theme, so all default layouts and features remain available while adding DevelopmentSeed-specific customizations.
