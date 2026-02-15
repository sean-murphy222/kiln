# CHONK Animation Specifications

A list of animations needed for the CHONK UI, with sizes and descriptions.

## Logo & Branding

| Animation | Size | Filename | Description |
|-----------|------|----------|-------------|
| Main Logo | 128x128px | `logo-main.png` | Fat cat mascot, idle animation (breathing/blinking) |
| Logo Small | 32x32px | `logo-small.png` | For toolbar/favicon, simplified version |
| Splash Screen | 256x256px | `splash.png` | Cat eating/chomping animation for app startup |

## Status & Feedback

| Animation | Size | Filename | Description |
|-----------|------|----------|-------------|
| Loading/Processing | 48x48px | `loading.png` | Cat chewing/munching - for document processing |
| Chunking in Progress | 64x64px | `chunking.png` | Cat slicing/chopping - during rechunk operations |
| Search/Thinking | 48x48px | `searching.png` | Cat with magnifying glass or thinking pose |
| Success | 48x48px | `success.png` | Happy cat, thumbs up, or satisfied expression |
| Error | 48x48px | `error.png` | Confused/concerned cat |
| Empty State | 128x128px | `empty.png` | Sleepy/waiting cat - "No documents yet" |

## Actions

| Animation | Size | Filename | Description |
|-----------|------|----------|-------------|
| Upload/Drop | 96x96px | `upload.png` | Cat catching or grabbing - for drag-drop zone |
| Export | 48x48px | `export.png` | Cat pushing/delivering a package |
| Save | 32x32px | `save.png` | Quick cat nod or checkmark |
| Merge Chunks | 48x48px | `merge.png` | Cat pushing things together |
| Split Chunk | 48x48px | `split.png` | Cat with scissors or karate chop |

## Quality Indicators

| Animation | Size | Filename | Description |
|-----------|------|----------|-------------|
| High Quality | 24x24px | `quality-high.png` | Happy/content cat face (static or subtle) |
| Medium Quality | 24x24px | `quality-medium.png` | Neutral cat face |
| Low Quality | 24x24px | `quality-low.png` | Concerned cat face |

---

## Technical Specifications

### Format
- **Preferred**: PNG sprite sheets (horizontal layout)
- **Alternative**: GIF or APNG
- Sprite sheets give better control over animation timing and looping

### Sprite Sheet Layout
```
┌────────┬────────┬────────┬────────┐
│ Frame1 │ Frame2 │ Frame3 │ Frame4 │  ← Single row, left to right
└────────┴────────┴────────┴────────┘
```

### Frame Rate
- **Recommended**: 8-12 fps
- Idle animations: 4-6 fps (slower, subtle)
- Action animations: 10-12 fps (snappier)

### Frame Count Guidelines
- Idle/breathing: 4-8 frames
- Simple actions (save, success): 4-6 frames
- Complex actions (chunking, upload): 8-12 frames
- Splash screen: 12-16 frames

### Color Palette
Match the CHONK 8-bit theme:

| Color | Hex | Usage |
|-------|-----|-------|
| Black | `#0a0a0f` | Outlines, shadows |
| Dark Purple | `#1a1a2e` | Background elements |
| Slate | `#4a4a6a` | Secondary elements |
| Gray | `#8888aa` | Neutral tones |
| Light | `#ccccdd` | Highlights |
| White | `#eeeef5` | Brightest highlights |
| Purple (Primary) | `#9d4edd` | Accent, interactive |
| Teal | `#2dd4bf` | Secondary accent |
| Green | `#22c55e` | Success states |
| Yellow/Warning | `#f59e0b` | Warning states |
| Red/Error | `#ef4444` | Error states |

### Style Guidelines
- Chunky pixel art aesthetic
- 1-2px outlines (dark purple or black)
- Minimal anti-aliasing (keep it crisp)
- Limited color palette (8-16 colors per sprite)
- Exaggerated, cute proportions for the cat

---

## File Organization

Place all animation files in:
```
ui/src/assets/animations/
├── logo-main.png
├── logo-small.png
├── splash.png
├── loading.png
├── chunking.png
├── searching.png
├── success.png
├── error.png
├── empty.png
├── upload.png
├── export.png
├── save.png
├── merge.png
├── split.png
├── quality-high.png
├── quality-medium.png
└── quality-low.png
```

---

## Priority Order

### Must Have (MVP)
1. Main Logo (128x128)
2. Logo Small (32x32)
3. Loading/Processing (48x48)
4. Empty State (128x128)
5. Error (48x48)

### Nice to Have
6. Upload/Drop (96x96)
7. Success (48x48)
8. Chunking in Progress (64x64)

### Polish
9. Splash Screen (256x256)
10. All remaining animations
