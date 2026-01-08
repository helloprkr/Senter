# Electron Transparent Window Configuration

## Overview
This document describes the configuration required to achieve a truly transparent, free-floating window in Electron for macOS.

## Key Changes Made

### 1. Electron BrowserWindow (`electron/main.js`)

**Removed:**
- `vibrancy: 'under-window'` - This creates macOS frosted glass effect which overrides transparency
- `visualEffectState: 'active'` - Related to vibrancy, not needed for true transparency

**Kept/Added:**
- `transparent: true` - Enables window transparency
- `backgroundColor: '#00000000'` - Fully transparent background (RGBA with 00 alpha)
- `hasShadow: false` - Removes window shadow for cleaner floating effect
- `frame: false` - Removes window frame
- `titleBarStyle: 'hiddenInset'` - Hidden title bar with inset traffic lights

### 2. Next.js Layout (`app/layout.tsx`)

Added inline styles to ensure transparency:
```tsx
<html lang="en" className="dark" style={{ background: 'transparent' }}>
  <body className="antialiased" style={{ background: 'transparent' }}>
```

### 3. Global Styles (`app/globals.css`)

Ensured html and body have transparent backgrounds with `!important` to override any Tailwind utilities:
```css
html,
body {
  background: transparent !important;
  background-color: transparent !important;
}

html.dark,
body {
  background: transparent !important;
  background-color: transparent !important;
}
```

## How It Works

1. **Electron Layer**: The BrowserWindow is configured with `transparent: true` and no vibrancy effects
2. **CSS Layer**: All root elements (html, body) have transparent backgrounds
3. **App Layer**: The main app component uses `bg-brand-light/[0.08]` which provides a semi-transparent container while letting the desktop show through around it

## Troubleshooting

If the window is not transparent:
1. Verify no `vibrancy` option is set in BrowserWindow
2. Check that no Tailwind `bg-background` class is applied to html/body
3. Ensure the CSS `!important` overrides are in place
4. Restart the Electron app completely (not just hot reload)

## Notes

- The app container itself (`liquid-glass-app.tsx`) uses `bg-brand-light/[0.08] backdrop-blur-3xl` for a frosted glass effect on the content area only
- The window chrome is fully transparent, allowing the desktop to show through
