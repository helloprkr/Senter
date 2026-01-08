# SenterUI - Comprehensive Specification Document
## Local macOS AI Application UI

---

## Table of Contents
1. [Brand Identity & Design System](#brand-identity--design-system)
2. [Application Architecture Overview](#application-architecture-overview)
3. [Core UI Components](#core-ui-components)
4. [Panel System & Animation Behavior](#panel-system--animation-behavior)
5. [Feature Specifications](#feature-specifications)
6. [Local AI Integration Guidelines](#local-ai-integration-guidelines)
7. [macOS-Specific Implementation](#macos-specific-implementation)
8. [Technical Stack Recommendations](#technical-stack-recommendations)

---

## Brand Identity & Design System

### Color Palette
```css
Primary (Dark Green):    #123524  /* Accents, borders, hover states */
Background (Deep Black): #081710  /* Main background, dark surfaces */
Light (Off-White):       #FAFAFA  /* Text, light UI elements */
Accent (Coral Red):      #D64933  /* Interactive elements, CTAs, highlights */
```

### Color Usage Guidelines
- **Primary (#123524)**: Use for:
  - Subtle background accents on glass panels
  - Border highlights on hover states
  - Secondary button backgrounds
  - Navigation item hover states
  - Ambient lighting effects (with low opacity)

- **Background (#081710)**: Use for:
  - Main application background
  - Dark panel backgrounds
  - Sidebar backgrounds
  - Overlay backgrounds

- **Light (#FAFAFA)**: Use for:
  - All body text
  - Icon colors (default state)
  - Input placeholder text (with opacity)
  - Border highlights on glass elements
  - Light UI component text

- **Accent (#D64933)**: Use for:
  - Active states on buttons
  - "Send" button background
  - Selected/active items
  - Interactive element highlights
  - Focus indicators
  - Attention-grabbing elements

### Typography System

**Font Family**: Manrope (Variable Font)
- **Source**: Custom font file `Manrope-VariableFont_wght.ttf`
- **Weight Range**: 200 (ExtraLight) to 800 (ExtraBold)

**Font Weights Used**:
```css
font-extralight: 200  /* Subtle UI elements */
font-light:      300  /* Body text, secondary information */
font-normal:     400  /* Standard text */
font-medium:     500  /* Headings, emphasized text */
font-semibold:   600  /* Strong emphasis, buttons */
font-bold:       700  /* Titles, important labels */
```

**Typography Scale**:
```css
text-xs:   0.75rem (12px)  /* Timestamps, meta info */
text-sm:   0.875rem (14px) /* Secondary text, captions */
text-base: 1rem (16px)     /* Body text, navigation items */
text-lg:   1.125rem (18px) /* Section headings */
text-xl:   1.25rem (20px)  /* Panel titles */
text-2xl:  1.5rem (24px)   /* Main headings */
```

**Letter Spacing**:
- Default body text: `tracking-normal` (0)
- Headings and labels: `tracking-wide` (0.025em)
- All-caps text: `tracking-wider` (0.05em)

### Liquid Glass Design System

**Core Glass Effect**:
```css
backdrop-blur-3xl:     blur(64px)
backdrop-blur-2xl:     blur(40px)
backdrop-blur-xl:      blur(24px)

/* Standard glass panel */
background: rgba(250, 250, 250, 0.08)  /* white/[0.08] */
backdrop-filter: blur(64px)
border: 1px solid rgba(250, 250, 250, 0.1)

/* Inner glow border effect */
box-shadow: 
  inset 0 1px 0 0 rgba(250, 250, 250, 0.1),
  0 0 40px rgba(18, 53, 36, 0.2)
```

**Glass Panel Hierarchy**:
1. **Background Layer**: Pure black (#081710) with ambient lighting
2. **Primary Glass Panels**: `bg-white/[0.08]` with `backdrop-blur-3xl`
3. **Secondary Glass Elements**: `bg-white/[0.05]` with `backdrop-blur-2xl`
4. **Interactive Glass Elements**: Transitions to primary color on hover
5. **Active States**: Accent color (#D64933) with increased opacity

**Ambient Lighting System**:
```css
/* Primary ambient light (top-left) */
position: fixed;
width: 600px;
height: 600px;
background: radial-gradient(circle, rgba(18, 53, 36, 0.3), transparent);
blur: 120px;

/* Accent ambient light (bottom-right) */
position: fixed;
width: 400px;
height: 400px;
background: radial-gradient(circle, rgba(214, 73, 51, 0.15), transparent);
blur: 100px;
```

### Icon System

**Icon Library**: Lucide React
**Default Icon Styling**:
```tsx
<Icon 
  className="w-5 h-5"           // 20px × 20px
  strokeWidth={1.5}             // Thin, modern strokes
  color="#FAFAFA"               // Light color by default
/>
```

**Icon Sizes**:
- Small icons: `w-4 h-4` (16px) - Meta actions, inline icons
- Standard icons: `w-5 h-5` (20px) - Navigation, buttons
- Large icons: `w-6 h-6` (24px) - Primary actions, headers

**Icon Color States**:
- Default: `#FAFAFA` (Light color)
- Hover: `#FAFAFA` with increased opacity or accent color
- Active: `#D64933` (Accent color)
- Disabled: `#FAFAFA` with 40% opacity

---

## Application Architecture Overview

### Application Window Structure

```
┌─────────────────────────────────────────────────────┐
│  macOS Application Window (Frameless/Transparent)   │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │   Ambient Lighting Layer (Fixed Background)    │ │
│  │                                                 │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │  Context Menu Panel (Expands UPWARD)    │  │ │
│  │  │  - Appears above input bar              │  │ │
│  │  │  - Animation: translateY(-100%) → 0%    │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                 │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │ ┌─────────┐  Main Sidebar (FIXED)       │  │ │
│  │  │ │ SENTER  │  - Always visible           │  │ │
│  │  │ └─────────┘  - Contains navigation      │  │ │
│  │  │              - Search bar               │  │ │
│  │  │              - Menu items               │  │ │
│  │  │  [Search]   - Settings                  │  │ │
│  │  │  [Nav...]   - User profile              │  │ │
│  │  │              - Does NOT move            │  │ │
│  │  │              - Z-index: 20              │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                 │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │  Floating Input Bar (FIXED AT BOTTOM)   │  │ │
│  │  │  - Always visible                       │  │ │
│  │  │  - Z-index: 50 (highest)                │  │ │
│  │  │  - Contains all primary controls        │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                 │ │
│  │  ┌──────────────────────────────────────────┐  │ │
│  │  │  Chat View Panel (Expands DOWNWARD)     │  │ │
│  │  │  - Appears below input bar              │  │ │
│  │  │  - Animation: translateY(100%) → 0%     │  │ │
│  │  └──────────────────────────────────────────┘  │ │
│  │                                                 │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Z-Index Layering System

```css
z-index: 0   - Background ambient lighting
z-index: 10  - Context Menu Panel
z-index: 20  - Main Sidebar (Fixed)
z-index: 30  - Chat View Panel
z-index: 50  - Floating Input Bar (Always on top)
```

**Critical Concept**: The Main Sidebar and Floating Input Bar are **FIXED** and never move. The Context Menu and Chat View panels slide in/out independently, appearing above/below the input bar respectively.

---

## Core UI Components

### 1. Main Sidebar (Fixed Component)

**Dimensions**:
- Width: `368px` (23rem)
- Height: `100vh` (Full viewport height)
- Position: `fixed` at left edge
- Z-index: `20`

**Structure**:
```tsx
<div className="fixed left-0 top-0 h-screen w-[368px] z-20">
  {/* Glass panel container */}
  <div className="h-full bg-white/[0.08] backdrop-blur-3xl border-r border-white/10">
    
    {/* Header Section */}
    <div className="flex items-center justify-between p-6">
      <div className="flex items-center gap-3">
        <MessageCircle /> {/* Icon: 24×24px */}
        <h1 className="text-xl font-medium">SENTER</h1>
      </div>
      <Button variant="ghost" size="icon">
        <X className="w-5 h-5" />
      </Button>
    </div>

    {/* Search Bar */}
    <div className="px-6 mb-6">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2" />
        <input 
          placeholder="Search (⌘K)"
          className="w-full bg-white/5 rounded-xl py-3 pl-10 pr-4"
        />
      </div>
    </div>

    {/* Navigation Items */}
    <nav className="flex-1 px-4 space-y-1">
      {/* Each nav item is a button with icon + text */}
    </nav>

    {/* User Profile (Bottom) */}
    <div className="p-4 border-t border-white/10">
      {/* User info with avatar */}
    </div>
  </div>
</div>
```

**Visual Specifications**:
- Background: `rgba(250, 250, 250, 0.08)` with `backdrop-blur(64px)`
- Border: `1px solid rgba(250, 250, 250, 0.1)` on right edge
- Inner glow: `inset 0 1px 0 0 rgba(250, 250, 250, 0.1)`
- Outer glow: `0 0 40px rgba(18, 53, 36, 0.2)`

**Navigation Item Styling**:
```css
/* Default state */
background: transparent
color: #FAFAFA
padding: 12px 16px
border-radius: 12px
transition: all 200ms ease-in-out

/* Hover state */
background: rgba(18, 53, 36, 0.3)  /* Primary color */
transform: translateX(4px)

/* Active state */
background: rgba(214, 73, 51, 0.2)  /* Accent color */
border-left: 2px solid #D64933
```

### 2. Floating Input Bar (Fixed Component)

**Dimensions**:
- Width: `calc(100vw - 368px - 96px)` (Full width minus sidebar and margins)
- Height: Dynamic based on content (min: `64px`)
- Position: `fixed` at bottom, centered
- Bottom offset: `32px`
- Z-index: `50`

**Structure**:
```tsx
<div className="fixed bottom-8 z-50" style={{
  left: '368px',  // Sidebar width
  width: 'calc(100vw - 368px - 96px)',
  marginLeft: '48px',
  marginRight: '48px'
}}>
  <div className="bg-white/[0.08] backdrop-blur-3xl rounded-2xl border border-white/10">
    <div className="flex items-center gap-3 p-4">
      
      {/* Left Controls */}
      <Button variant="ghost" size="icon">
        <FileText className="w-5 h-5" />
      </Button>

      {/* Text Input (Expandable) */}
      <textarea 
        placeholder="Type a message..."
        className="flex-1 bg-transparent resize-none"
        rows={1}
        onInput={(e) => {
          e.target.style.height = 'auto';
          e.target.style.height = e.target.scrollHeight + 'px';
        }}
      />

      {/* Right Controls */}
      <Button variant="ghost" size="icon">
        <Mic className="w-5 h-5" />
      </Button>
      <Button variant="ghost" size="icon">
        <Camera className="w-5 h-5" />
      </Button>
      <Button className="bg-[#D64933]">
        <Send className="w-5 h-5" />
      </Button>
      <Button variant="ghost" size="icon">
        <MessageSquare className="w-5 h-5" />
      </Button>
    </div>
  </div>
</div>
```

**Button Specifications**:

**Context Menu Button (FileText icon)**:
- Position: Leftmost button
- Function: Toggles Context Menu panel visibility
- Visual State Changes:
  - Default: `bg-white/5` with light icon
  - Hover: `bg-primary/30` (#123524 with 30% opacity)
  - Active (menu open): `bg-primary/40` with accent border

**Microphone Button (Mic icon)**:
- Position: Right side, before camera
- Function: Activates voice input recording
- Visual State Changes:
  - Default: Standard glass button
  - Active (recording): Rainbow animated border
    ```css
    @keyframes rainbow-border {
      0%, 100% { border-color: #FF0080; }
      16.66%   { border-color: #FF8C00; }
      33.33%   { border-color: #FFD700; }
      50%      { border-color: #00FF00; }
      66.66%   { border-color: #00BFFF; }
      83.33%   { border-color: #8A2BE2; }
    }
    animation: rainbow-border 3s linear infinite;
    border-width: 2px;
    ```

**Camera Button (Camera icon)**:
- Position: Right side, before send
- Function: Captures screenshot or opens camera
- Visual State Changes:
  - Default: Standard glass button
  - Pressed: Scale animation + brief accent color flash
  - Returns to default after capture

**Send Button (Send icon)**:
- Position: Right side, after camera
- Function: Submits the message/query
- Visual Specifications:
  - Background: `#D64933` (Accent color)
  - Hover: Darker accent `#B83F2B` with scale(1.05)
  - Active: Scale(0.95) for tactile feedback
  - Icon color: `#FAFAFA` (white)

**Chat History Button (MessageSquare icon)**:
- Position: Rightmost button
- Function: Toggles Chat View panel visibility
- Visual State Changes:
  - Default: `bg-white/5` with light icon
  - Hover: `bg-primary/30`
  - Active (chat open): `bg-primary/40` with accent border

**Input Text Area Behavior**:
- Default height: 1 row (approximately 24px)
- Max height: 5 rows (approximately 120px)
- Auto-expands as user types
- Keyboard shortcuts:
  - `Enter`: Send message
  - `Shift + Enter`: New line
- Placeholder text: "Type a message..." with 60% opacity

### 3. Context Menu Panel (Expands Upward)

**Dimensions**:
- Width: Same as Floating Input Bar
- Height: `600px` when expanded
- Position: `fixed` positioned ABOVE the input bar
- Animation Origin: Bottom edge (expands upward)

**Position Calculation**:
```tsx
// Calculate position to appear directly above input bar
const inputBarBottom = 32px;  // Input bar's bottom offset
const inputBarHeight = 64px;  // Approximate input bar height
const panelBottom = inputBarBottom + inputBarHeight + 8px;  // 8px gap

<div className="fixed z-10" style={{
  left: '368px',
  width: 'calc(100vw - 368px - 96px)',
  bottom: `${panelBottom}px`,
  height: isOpen ? '600px' : '0px',
  marginLeft: '48px',
  marginRight: '48px'
}}>
```

**Animation Behavior**:
```tsx
// Using Framer Motion
<motion.div
  initial={{ height: 0, opacity: 0 }}
  animate={{ 
    height: isContextMenuOpen ? 600 : 0,
    opacity: isContextMenuOpen ? 1 : 0
  }}
  transition={{ 
    duration: 0.3,
    ease: [0.16, 1, 0.3, 1]  // Custom easing for smooth feel
  }}
  style={{ overflow: 'hidden' }}
>
```

**Structure**:
```tsx
<div className="h-full bg-white/[0.08] backdrop-blur-3xl rounded-2xl border border-white/10">
  <div className="flex h-full">
    
    {/* Filter Sidebar (Left) */}
    <div className="w-48 border-r border-white/10 p-4">
      <h3 className="text-sm font-medium mb-4">Context Sources</h3>
      <div className="space-y-2">
        <Button variant={filter === 'all' ? 'default' : 'ghost'}>
          All
        </Button>
        <Button variant={filter === 'web' ? 'default' : 'ghost'}>
          Web
        </Button>
        <Button variant={filter === 'files' ? 'default' : 'ghost'}>
          Files
        </Button>
        <Button variant={filter === 'clipboard' ? 'default' : 'ghost'}>
          Clipboard
        </Button>
      </div>
    </div>

    {/* Context Items (Right) */}
    <div className="flex-1 flex flex-col">
      
      {/* Search Bar */}
      <div className="p-4 border-b border-white/10">
        <input 
          placeholder="Search context items..."
          className="w-full bg-white/5 rounded-lg py-2 px-3"
        />
      </div>

      {/* Scrollable Items List */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-2">
          {/* Context Item Cards */}
          <div className="bg-white/5 rounded-lg p-3 hover:bg-white/10">
            <div className="flex items-start gap-3">
              <Icon className="w-5 h-5 mt-1" />
              <div className="flex-1 min-w-0">
                <h4 className="font-medium text-sm">Item Title</h4>
                <p className="text-xs text-white/60 truncate">
                  Item description or preview text...
                </p>
              </div>
              <Button variant="ghost" size="icon">
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
```

**Context Item Types & Icons**:
```tsx
const contextItemIcons = {
  web: Globe,       // Web page or URL
  file: FileText,   // Document or file
  clipboard: Copy,  // Clipboard content
  image: Image,     // Image file
  code: Code,       // Code snippet
};
```

**Filter Button Styling**:
```css
/* Inactive filter */
background: transparent
color: #FAFAFA
opacity: 0.7

/* Active filter */
background: rgba(18, 53, 36, 0.4)
color: #FAFAFA
opacity: 1
border-left: 2px solid #D64933
```

### 4. Chat View Panel (Expands Downward)

**Dimensions**:
- Width: Same as Floating Input Bar
- Height: `calc(100vh - input bar bottom - input bar height - 104px)`
- Position: `fixed` positioned BELOW the input bar
- Animation Origin: Top edge (expands downward)

**Position Calculation**:
```tsx
const inputBarBottom = 32px;
const inputBarHeight = 64px;
const panelTop = inputBarBottom + inputBarHeight + 8px;

<div className="fixed z-30" style={{
  left: '368px',
  width: 'calc(100vw - 368px - 96px)',
  top: `calc(100vh - ${inputBarBottom}px - ${inputBarHeight}px - 8px)`,
  height: isOpen ? 'calc(100vh - 200px)' : '0px',
  marginLeft: '48px',
  marginRight: '48px'
}}>
```

**Animation Behavior**:
```tsx
<motion.div
  initial={{ height: 0, opacity: 0 }}
  animate={{ 
    height: isChatOpen ? 'calc(100vh - 200px)' : 0,
    opacity: isChatOpen ? 1 : 0
  }}
  transition={{ 
    duration: 0.3,
    ease: [0.16, 1, 0.3, 1]
  }}
  style={{ overflow: 'hidden' }}
>
```

**Structure**:
```tsx
<div className="h-full bg-white/[0.08] backdrop-blur-3xl rounded-2xl border border-white/10">
  <div className="flex h-full">
    
    {/* Chat Messages (Left, Primary) */}
    <div className="flex-1 flex flex-col">
      
      {/* Chat Header */}
      <div className="p-4 border-b border-white/10">
        <h3 className="text-lg font-medium">Conversation</h3>
      </div>

      {/* Messages Container (Scrollable) */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        
        {/* User Message */}
        <div className="flex justify-end">
          <div className="bg-[#D64933]/20 rounded-2xl rounded-tr-sm px-4 py-3 max-w-[70%]">
            <p className="text-sm">User message text here</p>
            <span className="text-xs text-white/50 mt-1">10:24 AM</span>
          </div>
        </div>

        {/* Bot Message */}
        <div className="flex justify-start">
          <div className="bg-white/10 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[70%]">
            <p className="text-sm">AI response text here</p>
            <span className="text-xs text-white/50 mt-1">10:24 AM</span>
          </div>
        </div>
      </div>
    </div>

    {/* Sidebar (Right) */}
    <div className="w-80 border-l border-white/10 flex flex-col">
      
      {/* Tab Navigation */}
      <div className="flex border-b border-white/10">
        <button className="flex-1 py-3 text-sm font-medium">
          Chat History
        </button>
        <button className="flex-1 py-3 text-sm font-medium">
          Journal
        </button>
        <button className="flex-1 py-3 text-sm font-medium">
          Tasks
        </button>
      </div>

      {/* Tab Content (Scrollable) */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Tab-specific content */}
      </div>
    </div>
  </div>
</div>
```

**Message Styling**:
```css
/* User messages (right-aligned) */
background: rgba(214, 73, 51, 0.2)  /* Accent color with transparency */
border-radius: 16px 16px 4px 16px   /* Rounded with chat tail */
align-self: flex-end
max-width: 70%

/* Bot messages (left-aligned) */
background: rgba(250, 250, 250, 0.1)  /* Glass effect */
border-radius: 16px 16px 16px 4px     /* Rounded with chat tail */
align-self: flex-start
max-width: 70%

/* Timestamp */
font-size: 0.75rem (12px)
color: rgba(250, 250, 250, 0.5)
margin-top: 4px
```

**Tab Navigation Styling**:
```css
/* Inactive tab */
background: transparent
color: rgba(250, 250, 250, 0.6)
border-bottom: 2px solid transparent

/* Active tab */
background: rgba(18, 53, 36, 0.2)
color: #FAFAFA
border-bottom: 2px solid #D64933
```

---

## Panel System & Animation Behavior

### Core Animation Principles

**1. Independent Panel Movement**
- Main Sidebar: **NEVER MOVES** (fixed at left edge)
- Floating Input Bar: **NEVER MOVES** (fixed at bottom center)
- Context Menu: Expands/collapses **UPWARD** from input bar
- Chat View: Expands/collapses **DOWNWARD** from input bar

**2. Animation Timing**
```typescript
const animationConfig = {
  duration: 0.3,  // 300ms - feels responsive, not rushed
  ease: [0.16, 1, 0.3, 1],  // Custom cubic-bezier for smooth, natural feel
};
```

**3. Stagger Effect (Optional Enhancement)**
For multiple items animating in (like context items or messages):
```typescript
const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,  // 50ms delay between each child
    }
  }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};
```

### Panel State Management

**State Variables Needed**:
```typescript
const [isContextMenuOpen, setIsContextMenuOpen] = useState(false);
const [isChatViewOpen, setIsChatViewOpen] = useState(false);
const [isMicActive, setIsMicActive] = useState(false);
const [contextFilter, setContextFilter] = useState<'all' | 'web' | 'files' | 'clipboard'>('all');
const [activeTab, setActiveTab] = useState<'history' | 'journal' | 'tasks'>('history');
```

**Toggle Functions**:
```typescript
const toggleContextMenu = () => {
  setIsContextMenuOpen(!isContextMenuOpen);
  // Optionally close chat view if both shouldn't be open simultaneously
  if (!isContextMenuOpen && isChatViewOpen) {
    setIsChatViewOpen(false);
  }
};

const toggleChatView = () => {
  setIsChatViewOpen(!isChatViewOpen);
  // Optionally close context menu
  if (!isChatViewOpen && isContextMenuOpen) {
    setIsContextMenuOpen(false);
  }
};
```

### Animation Implementation (Framer Motion)

**Context Menu Animation**:
```tsx
<AnimatePresence>
  {isContextMenuOpen && (
    <motion.div
      className="fixed z-10"
      style={{
        left: '368px',
        bottom: '104px',  // Above input bar
        width: 'calc(100vw - 464px)',
        marginLeft: '48px',
      }}
      initial={{ 
        height: 0, 
        opacity: 0,
        y: 20  // Start slightly lower
      }}
      animate={{ 
        height: 600, 
        opacity: 1,
        y: 0
      }}
      exit={{ 
        height: 0, 
        opacity: 0,
        y: 20  // Exit downward
      }}
      transition={{ 
        duration: 0.3,
        ease: [0.16, 1, 0.3, 1]
      }}
    >
      {/* Context Menu Content */}
    </motion.div>
  )}
</AnimatePresence>
```

**Chat View Animation**:
```tsx
<AnimatePresence>
  {isChatViewOpen && (
    <motion.div
      className="fixed z-30"
      style={{
        left: '368px',
        top: '104px',  // Below input bar (but positioned from top for downward expansion)
        width: 'calc(100vw - 464px)',
        marginLeft: '48px',
      }}
      initial={{ 
        height: 0, 
        opacity: 0,
        y: -20  // Start slightly higher
      }}
      animate={{ 
        height: 'calc(100vh - 200px)', 
        opacity: 1,
        y: 0
      }}
      exit={{ 
        height: 0, 
        opacity: 0,
        y: -20  // Exit upward
      }}
      transition={{ 
        duration: 0.3,
        ease: [0.16, 1, 0.3, 1]
      }}
    >
      {/* Chat View Content */}
    </motion.div>
  )}
</AnimatePresence>
```

### Visual Feedback During Animation

**Loading States**:
```tsx
{isContextMenuOpen && (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ delay: 0.15 }}  // Appear mid-animation
    className="absolute top-4 right-4"
  >
    <Loader2 className="w-5 h-5 animate-spin text-white/50" />
  </motion.div>
)}
```

**Backdrop Overlay (Optional)**:
```tsx
<AnimatePresence>
  {(isContextMenuOpen || isChatViewOpen) && (
    <motion.div
      className="fixed inset-0 bg-black/20 backdrop-blur-sm z-5"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={() => {
        setIsContextMenuOpen(false);
        setIsChatViewOpen(false);
      }}
    />
  )}
</AnimatePresence>
```

---

## Feature Specifications

### Feature 1: Search Bar (Main Sidebar)

**Purpose**: Global search across the application
**Location**: Main Sidebar, below header
**Keyboard Shortcut**: `⌘K` (macOS) or `Ctrl+K` (Windows/Linux)

**Implementation**:
```tsx
const [searchQuery, setSearchQuery] = useState('');
const [isSearchFocused, setIsSearchFocused] = useState(false);

// Listen for keyboard shortcut
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      searchInputRef.current?.focus();
    }
  };
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, []);

<div className="relative">
  <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 transition-colors ${
    isSearchFocused ? 'text-[#D64933]' : 'text-white/60'
  }`} />
  <input
    ref={searchInputRef}
    type="text"
    placeholder="Search (⌘K)"
    value={searchQuery}
    onChange={(e) => setSearchQuery(e.target.value)}
    onFocus={() => setIsSearchFocused(true)}
    onBlur={() => setIsSearchFocused(false)}
    className={`w-full bg-white/5 rounded-xl py-3 pl-10 pr-4 text-sm transition-all ${
      isSearchFocused ? 'bg-white/10 ring-2 ring-[#D64933]/50' : ''
    }`}
  />
</div>
```

**Search Behavior**:
- Searches across: Navigation items, chat history, files, context items
- Shows real-time results in a dropdown below the search bar
- `Escape` key closes search results
- Arrow keys navigate results
- `Enter` selects highlighted result

### Feature 2: Navigation Menu

**Purpose**: Primary navigation for different AI features
**Location**: Main Sidebar, between search and user profile

**Navigation Items**:
```typescript
const navigationItems = [
  { 
    icon: MessageCircle, 
    label: 'Conversation', 
    id: 'conversation',
    description: 'Chat with AI assistant'
  },
  { 
    icon: Lightbulb, 
    label: 'Smart Lights', 
    id: 'smart-lights',
    description: 'Control smart home lighting'
  },
  { 
    icon: Camera, 
    label: 'Vision', 
    id: 'vision',
    description: 'Image analysis and recognition'
  },
  { 
    icon: Search, 
    label: 'Research', 
    id: 'research',
    description: 'Web research and data gathering'
  },
  { 
    icon: Settings, 
    label: 'Settings', 
    id: 'settings',
    dropdown: true,  // Has submenu
    description: 'Application settings'
  },
];
```

**Implementation**:
```tsx
const [activeNav, setActiveNav] = useState('conversation');
const [settingsOpen, setSettingsOpen] = useState(false);

{navigationItems.map((item) => (
  <motion.button
    key={item.id}
    onClick={() => {
      if (item.dropdown) {
        setSettingsOpen(!settingsOpen);
      } else {
        setActiveNav(item.id);
        // Navigate to feature
      }
    }}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
      activeNav === item.id 
        ? 'bg-[#D64933]/20 border-l-2 border-[#D64933]' 
        : 'hover:bg-[#123524]/30 hover:translate-x-1'
    }`}
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
  >
    <item.icon className="w-5 h-5" strokeWidth={1.5} />
    <span className="text-sm font-light">{item.label}</span>
    {item.dropdown && (
      <ChevronDown className={`w-4 h-4 ml-auto transition-transform ${
        settingsOpen ? 'rotate-180' : ''
      }`} />
    )}
  </motion.button>
))}
```

**Settings Dropdown**:
```tsx
<AnimatePresence>
  {settingsOpen && (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: 'auto', opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      className="ml-4 mt-1 space-y-1 overflow-hidden"
    >
      <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-white/5 text-sm">
        Preferences
      </button>
      <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-white/5 text-sm">
        API Keys
      </button>
      <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-white/5 text-sm">
        Integrations
      </button>
    </motion.div>
  )}
</AnimatePresence>
```

### Feature 3: Context Menu System

**Purpose**: Manage contextual information for AI queries
**Trigger**: FileText button in input bar (leftmost button)
**Panel**: Expands upward from input bar

**Context Sources**:
1. **Web**: URLs, web page content, search results
2. **Files**: Local files, documents, code files
3. **Clipboard**: Copied text, images, code snippets
4. **All**: Combined view of all sources

**Context Item Structure**:
```typescript
interface ContextItem {
  id: string;
  type: 'web' | 'file' | 'clipboard' | 'image' | 'code';
  title: string;
  content: string;
  preview: string;  // Short preview text
  timestamp: Date;
  metadata?: {
    url?: string;       // For web items
    filePath?: string;  // For file items
    fileSize?: number;  // For file items
    mimeType?: string;  // For various types
  };
}
```

**Adding Context Items**:
```typescript
const addContextItem = (item: Omit<ContextItem, 'id' | 'timestamp'>) => {
  const newItem: ContextItem = {
    ...item,
    id: generateId(),
    timestamp: new Date(),
  };
  setContextItems([newItem, ...contextItems]);
};

// Example: Add URL from clipboard
const handleAddUrl = async () => {
  const clipboardText = await navigator.clipboard.readText();
  if (isValidUrl(clipboardText)) {
    addContextItem({
      type: 'web',
      title: 'Web Page',
      content: clipboardText,
      preview: clipboardText,
      metadata: { url: clipboardText }
    });
  }
};

// Example: Add file
const handleAddFile = async () => {
  const file = await openFileDialog();
  if (file) {
    addContextItem({
      type: 'file',
      title: file.name,
      content: await file.text(),
      preview: `${file.name} (${formatFileSize(file.size)})`,
      metadata: {
        filePath: file.path,
        fileSize: file.size,
        mimeType: file.type
      }
    });
  }
};
```

**Removing Context Items**:
```typescript
const removeContextItem = (id: string) => {
  setContextItems(contextItems.filter(item => item.id !== id));
};
```

**Context Filtering**:
```typescript
const filteredContextItems = contextItems.filter(item => {
  if (contextFilter === 'all') return true;
  return item.type === contextFilter;
});
```

**Search within Context**:
```typescript
const [contextSearch, setContextSearch] = useState('');

const searchedContextItems = filteredContextItems.filter(item =>
  item.title.toLowerCase().includes(contextSearch.toLowerCase()) ||
  item.preview.toLowerCase().includes(contextSearch.toLowerCase())
);
```

### Feature 4: Microphone Input with Rainbow Animation

**Purpose**: Voice input for AI queries
**Trigger**: Mic button in input bar
**Visual Feedback**: Rainbow animated border during recording

**State Management**:
```typescript
const [isMicActive, setIsMicActive] = useState(false);
const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
const [recordedAudio, setRecordedAudio] = useState<Blob | null>(null);
```

**Recording Implementation**:
```typescript
const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    setAudioStream(stream);
    setIsMicActive(true);

    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks: Blob[] = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      setRecordedAudio(audioBlob);
      // Send to transcription service
      transcribeAudio(audioBlob);
    };

    mediaRecorder.start();
    
    // Store mediaRecorder reference for stopping
    window.currentMediaRecorder = mediaRecorder;
  } catch (error) {
    console.error('Error accessing microphone:', error);
  }
};

const stopRecording = () => {
  if (window.currentMediaRecorder) {
    window.currentMediaRecorder.stop();
  }
  if (audioStream) {
    audioStream.getTracks().forEach(track => track.stop());
    setAudioStream(null);
  }
  setIsMicActive(false);
};
```

**Rainbow Border Animation**:
```css
/* In globals.css */
@keyframes rainbow-border {
  0%, 100% { 
    border-color: #FF0080; 
    box-shadow: 0 0 20px #FF0080;
  }
  16.66% { 
    border-color: #FF8C00;
    box-shadow: 0 0 20px #FF8C00;
  }
  33.33% { 
    border-color: #FFD700;
    box-shadow: 0 0 20px #FFD700;
  }
  50% { 
    border-color: #00FF00;
    box-shadow: 0 0 20px #00FF00;
  }
  66.66% { 
    border-color: #00BFFF;
    box-shadow: 0 0 20px #00BFFF;
  }
  83.33% { 
    border-color: #8A2BE2;
    box-shadow: 0 0 20px #8A2BE2;
  }
}

.rainbow-border-active {
  animation: rainbow-border 3s linear infinite;
  border-width: 2px;
  border-style: solid;
}
```

**Button Implementation**:
```tsx
<Button
  variant="ghost"
  size="icon"
  onClick={isMicActive ? stopRecording : startRecording}
  className={`relative ${isMicActive ? 'rainbow-border-active' : ''}`}
>
  <Mic className={`w-5 h-5 ${isMicActive ? 'text-[#D64933]' : ''}`} />
  {isMicActive && (
    <span className="absolute -top-1 -right-1 w-3 h-3 bg-[#D64933] rounded-full animate-pulse" />
  )}
</Button>
```

### Feature 5: Camera/Screenshot Capture

**Purpose**: Capture screenshots or images for visual analysis
**Trigger**: Camera button in input bar
**Behavior**: Brief visual feedback on capture

**Implementation**:
```typescript
const captureScreenshot = async () => {
  try {
    // For macOS, you might use a native screenshot API
    // This is a browser-based example
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { mediaSource: 'screen' }
    });

    const video = document.createElement('video');
    video.srcObject = stream;
    await video.play();

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx?.drawImage(video, 0, 0);

    stream.getTracks().forEach(track => track.stop());

    canvas.toBlob((blob) => {
      if (blob) {
        // Add to context items
        addContextItem({
          type: 'image',
          title: 'Screenshot',
          content: URL.createObjectURL(blob),
          preview: 'Captured screenshot',
          metadata: {
            mimeType: 'image/png'
          }
        });
      }
    });
  } catch (error) {
    console.error('Error capturing screenshot:', error);
  }
};
```

**Button with Flash Animation**:
```tsx
const [isFlashing, setIsFlashing] = useState(false);

const handleCameraClick = async () => {
  setIsFlashing(true);
  await captureScreenshot();
  setTimeout(() => setIsFlashing(false), 200);
};

<Button
  variant="ghost"
  size="icon"
  onClick={handleCameraClick}
  className={`transition-all ${isFlashing ? 'scale-95 bg-white/20' : ''}`}
>
  <Camera className="w-5 h-5" />
</Button>

{/* Flash overlay */}
<AnimatePresence>
  {isFlashing && (
    <motion.div
      className="fixed inset-0 bg-white z-[100] pointer-events-none"
      initial={{ opacity: 0.8 }}
      animate={{ opacity: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    />
  )}
</AnimatePresence>
```

### Feature 6: Message Input & Submission

**Purpose**: Text input for AI queries with multi-line support
**Location**: Center of input bar
**Behavior**: Auto-expands up to 5 lines

**Implementation**:
```typescript
const [message, setMessage] = useState('');
const textareaRef = useRef<HTMLTextAreaElement>(null);

const handleInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
  const target = e.currentTarget;
  target.style.height = 'auto';
  target.style.height = `${Math.min(target.scrollHeight, 120)}px`;
  setMessage(target.value);
};

const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSendMessage();
  }
};

const handleSendMessage = async () => {
  if (!message.trim()) return;

  const userMessage = message;
  setMessage('');
  
  // Reset textarea height
  if (textareaRef.current) {
    textareaRef.current.style.height = 'auto';
  }

  // Add user message to chat
  addMessage({
    role: 'user',
    content: userMessage,
    timestamp: new Date()
  });

  // Send to AI backend
  const response = await sendToAI(userMessage, contextItems);
  
  // Add AI response to chat
  addMessage({
    role: 'assistant',
    content: response,
    timestamp: new Date()
  });
};

<textarea
  ref={textareaRef}
  value={message}
  onChange={handleInput}
  onKeyDown={handleKeyDown}
  placeholder="Type a message..."
  className="flex-1 bg-transparent border-none outline-none resize-none text-sm py-2 max-h-[120px]"
  rows={1}
  style={{ minHeight: '24px' }}
/>
```

### Feature 7: Chat View with History

**Purpose**: View conversation history and manage chats
**Trigger**: MessageSquare button in input bar (rightmost)
**Panel**: Expands downward from input bar

**Chat Message Structure**:
```typescript
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    model?: string;      // AI model used
    tokens?: number;     // Token count
    contextUsed?: string[];  // IDs of context items used
  };
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}
```

**Message Display**:
```tsx
const MessageBubble = ({ message }: { message: ChatMessage }) => {
  const isUser = message.role === 'user';
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`max-w-[70%] rounded-2xl px-4 py-3 ${
        isUser 
          ? 'bg-[#D64933]/20 rounded-tr-sm' 
          : 'bg-white/10 rounded-tl-sm'
      }`}>
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        <span className="text-xs text-white/50 mt-1 block">
          {formatTime(message.timestamp)}
        </span>
      </div>
    </motion.div>
  );
};
```

**Chat History Sidebar**:
```tsx
const ChatHistorySidebar = ({ sessions, activeSessionId, onSelectSession }: Props) => {
  return (
    <div className="space-y-2">
      {sessions.map(session => (
        <button
          key={session.id}
          onClick={() => onSelectSession(session.id)}
          className={`w-full text-left p-3 rounded-lg transition-all ${
            activeSessionId === session.id
              ? 'bg-[#123524]/40 border-l-2 border-[#D64933]'
              : 'hover:bg-white/5'
          }`}
        >
          <h4 className="text-sm font-medium truncate">{session.title}</h4>
          <p className="text-xs text-white/50 mt-1">
            {formatDate(session.updatedAt)}
          </p>
        </button>
      ))}
    </div>
  );
};
```

### Feature 8: User Profile Section

**Purpose**: Display user info and quick access to account settings
**Location**: Bottom of main sidebar
**Contains**: Avatar, name, status

**Implementation**:
```tsx
const UserProfile = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        className="w-full flex items-center gap-3 p-4 border-t border-white/10 hover:bg-white/5 transition-colors"
      >
        <div className="w-10 h-10 rounded-full bg-[#123524] flex items-center justify-center">
          <User className="w-5 h-5" />
        </div>
        <div className="flex-1 text-left">
          <p className="text-sm font-medium">Desktop User</p>
          <p className="text-xs text-white/60">Local Mode</p>
        </div>
        <ChevronUp className={`w-4 h-4 transition-transform ${
          isMenuOpen ? '' : 'rotate-180'
        }`} />
      </button>

      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute bottom-full left-4 right-4 mb-2 bg-white/10 backdrop-blur-xl rounded-xl border border-white/10 overflow-hidden"
          >
            <button className="w-full text-left px-4 py-3 hover:bg-white/5 text-sm">
              Profile Settings
            </button>
            <button className="w-full text-left px-4 py-3 hover:bg-white/5 text-sm">
              Switch Mode
            </button>
            <button className="w-full text-left px-4 py-3 hover:bg-white/5 text-sm text-[#D64933]">
              Sign Out
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
```

---

## Local AI Integration Guidelines

### Architecture Overview

This UI is designed to be a frontend for a **local AI agent system**. Here's how to integrate your AI backend:

```
┌─────────────────────────────────────────────────────┐
│                  macOS Application                   │
│  ┌────────────────────────────────────────────────┐ │
│  │         UI Layer (Electron/Tauri)              │ │
│  │  - Liquid Glass Interface                      │ │
│  │  - User Interactions                           │ │
│  │  - Visual Feedback                             │ │
│  └────────────┬───────────────────────────────────┘ │
│               │                                      │
│               │ IPC / WebSocket                      │
│               │                                      │
│  ┌────────────▼───────────────────────────────────┐ │
│  │      Application Backend (Node.js/Python)      │ │
│  │  - Message Routing                             │ │
│  │  - Context Management                          │ │
│  │  - State Management                            │ │
│  └────────────┬───────────────────────────────────┘ │
│               │                                      │
│               │ HTTP / gRPC                          │
│               │                                      │
│  ┌────────────▼───────────────────────────────────┐ │
│  │        Local AI Agent System                   │ │
│  │  - LLM Inference Engine (Ollama, LM Studio)   │ │
│  │  - RAG System (Vector DB)                     │ │
│  │  - Tool/Function Calling                      │ │
│  │  - Response Generation                        │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Message Flow

**1. User Sends Message**:
```typescript
// In UI Layer
const handleSendMessage = async () => {
  const payload = {
    message: userMessage,
    context: contextItems,
    sessionId: currentSessionId,
    settings: {
      model: selectedModel,
      temperature: 0.7,
      maxTokens: 2000
    }
  };

  // Send to backend via IPC (Electron) or invoke (Tauri)
  const response = await window.api.sendMessage(payload);
  
  // Handle response
  addMessageToChat(response);
};
```

**2. Backend Processes Request**:
```typescript
// In Backend Layer (example using Electron IPC)
ipcMain.handle('send-message', async (event, payload) => {
  try {
    // Extract context
    const contextData = await processContextItems(payload.context);
    
    // Prepare prompt with context
    const augmentedPrompt = `
      Context Information:
      ${contextData}
      
      User Query:
      ${payload.message}
    `;
    
    // Send to local AI
    const aiResponse = await sendToLocalAI({
      prompt: augmentedPrompt,
      model: payload.settings.model,
      temperature: payload.settings.temperature,
      maxTokens: payload.settings.maxTokens
    });
    
    // Return response to UI
    return {
      role: 'assistant',
      content: aiResponse.text,
      metadata: {
        model: aiResponse.model,
        tokens: aiResponse.tokensUsed,
        processingTime: aiResponse.time
      }
    };
  } catch (error) {
    return {
      role: 'system',
      content: `Error: ${error.message}`,
      metadata: { error: true }
    };
  }
});
```

**3. Local AI Processing**:
```python
# Example using Ollama (Python)
import ollama

def process_query(prompt, model='llama2', temperature=0.7, max_tokens=2000):
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            'temperature': temperature,
            'num_predict': max_tokens
        }
    )
    
    return {
        'text': response['response'],
        'model': model,
        'tokensUsed': response['eval_count'],
        'time': response['total_duration'] / 1e9  # Convert to seconds
    }
```

### Context Item Processing

**File Context**:
```typescript
const processFileContext = async (item: ContextItem) => {
  if (item.type === 'file') {
    // Read file content
    const content = await fs.readFile(item.metadata.filePath, 'utf-8');
    
    // For code files, include syntax highlighting info
    if (isCodeFile(item.metadata.filePath)) {
      return {
        type: 'code',
        language: detectLanguage(item.metadata.filePath),
        content: content,
        path: item.metadata.filePath
      };
    }
    
    return {
      type: 'document',
      content: content,
      name: item.title
    };
  }
};
```

**Web Context**:
```typescript
const processWebContext = async (item: ContextItem) => {
  if (item.type === 'web') {
    // Fetch and parse web content
    const response = await fetch(item.metadata.url);
    const html = await response.text();
    
    // Extract main content (remove nav, ads, etc.)
    const mainContent = extractMainContent(html);
    
    return {
      type: 'web',
      url: item.metadata.url,
      title: item.title,
      content: mainContent
    };
  }
};
```

**Image Context**:
```typescript
const processImageContext = async (item: ContextItem) => {
  if (item.type === 'image') {
    // For vision-enabled models
    const imageBuffer = await fs.readFile(item.content);
    const base64Image = imageBuffer.toString('base64');
    
    return {
      type: 'image',
      data: base64Image,
      mimeType: item.metadata.mimeType,
      description: item.preview
    };
  }
};
```

### Streaming Responses

For real-time AI response streaming:

```typescript
// In UI Layer
const handleStreamingMessage = async () => {
  const stream = await window.api.sendMessageStream(payload);
  
  let accumulatedResponse = '';
  
  for await (const chunk of stream) {
    accumulatedResponse += chunk;
    
    // Update UI in real-time
    updateMessageContent(messageId, accumulatedResponse);
  }
};

// In Backend Layer
ipcMain.handle('send-message-stream', async (event, payload) => {
  const stream = await aiClient.streamGenerate({
    prompt: augmentedPrompt,
    model: payload.settings.model
  });
  
  // Create async generator
  async function* generateChunks() {
    for await (const chunk of stream) {
      yield chunk.text;
    }
  }
  
  return generateChunks();
});
```

### RAG (Retrieval-Augmented Generation) Integration

```typescript
// Vector database integration for context retrieval
const retrieveRelevantContext = async (query: string) => {
  // Query vector database
  const results = await vectorDB.search({
    query: query,
    limit: 5,
    threshold: 0.7
  });
  
  return results.map(result => ({
    content: result.text,
    source: result.metadata.source,
    relevanceScore: result.score
  }));
};

// Augment prompt with retrieved context
const augmentPromptWithRAG = async (userMessage: string) => {
  const relevantDocs = await retrieveRelevantContext(userMessage);
  
  const contextSection = relevantDocs
    .map((doc, i) => `[${i + 1}] ${doc.content}\nSource: ${doc.source}`)
    .join('\n\n');
  
  return `
    Relevant Information:
    ${contextSection}
    
    User Query:
    ${userMessage}
    
    Please provide a response based on the relevant information above.
  `;
};
```

### Tool/Function Calling

For AI agents that can call functions:

```typescript
// Define available tools
const tools = {
  searchWeb: async (query: string) => {
    // Implement web search
    return await webSearch(query);
  },
  
  readFile: async (path: string) => {
    // Read local file
    return await fs.readFile(path, 'utf-8');
  },
  
  executeCode: async (code: string, language: string) => {
    // Execute code safely
    return await sandboxedCodeExecution(code, language);
  }
};

// Process tool calls from AI
const processToolCalls = async (toolCalls: ToolCall[]) => {
  const results = [];
  
  for (const call of toolCalls) {
    if (call.name in tools) {
      const result = await tools[call.name](...call.arguments);
      results.push({
        toolCall: call.id,
        result: result
      });
    }
  }
  
  return results;
};
```

### Session Persistence

```typescript
// Save chat session to local database
const saveChatSession = async (session: ChatSession) => {
  // Using SQLite or similar
  await db.sessions.upsert({
    where: { id: session.id },
    create: {
      id: session.id,
      title: session.title,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
      messages: JSON.stringify(session.messages)
    },
    update: {
      title: session.title,
      updatedAt: session.updatedAt,
      messages: JSON.stringify(session.messages)
    }
  });
};

// Load chat sessions on startup
const loadChatSessions = async () => {
  const sessions = await db.sessions.findMany({
    orderBy: { updatedAt: 'desc' }
  });
  
  return sessions.map(session => ({
    ...session,
    messages: JSON.parse(session.messages)
  }));
};
```

---

## macOS-Specific Implementation

### Window Configuration (Electron)

```typescript
// main.ts (Electron main process)
import { app, BrowserWindow } from 'electron';

const createWindow = () => {
  const mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    titleBarStyle: 'hiddenInset',  // macOS native title bar
    frame: false,                   // Frameless window
    transparent: true,              // Transparent background
    vibrancy: 'ultra-dark',        // macOS vibrancy effect
    visualEffectState: 'active',
    backgroundColor: '#00000000',   // Transparent
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  mainWindow.loadFile('index.html');
};

app.whenReady().then(createWindow);
```

### Window Configuration (Tauri)

```json
// tauri.conf.json
{
  "tauri": {
    "windows": [{
      "title": "SenterUI",
      "width": 1400,
      "height": 900,
      "minWidth": 1200,
      "minHeight": 700,
      "decorations": false,
      "transparent": true,
      "resizable": true,
      "fullscreen": false,
      "center": true
    }],
    "macOSPrivateApi": true
  }
}
```

### Native Menu Integration

```typescript
// Create macOS menu bar
import { Menu, app } from 'electron';

const template: MenuItemConstructorOptions[] = [
  {
    label: app.name,
    submenu: [
      { role: 'about' },
      { type: 'separator' },
      { role: 'services' },
      { type: 'separator' },
      { role: 'hide' },
      { role: 'hideOthers' },
      { role: 'unhide' },
      { type: 'separator' },
      { role: 'quit' }
    ]
  },
  {
    label: 'Edit',
    submenu: [
      { role: 'undo' },
      { role: 'redo' },
      { type: 'separator' },
      { role: 'cut' },
      { role: 'copy' },
      { role: 'paste' },
      { role: 'delete' },
      { role: 'selectAll' }
    ]
  },
  {
    label: 'View',
    submenu: [
      { role: 'reload' },
      { role: 'forceReload' },
      { role: 'toggleDevTools' },
      { type: 'separator' },
      { role: 'resetZoom' },
      { role: 'zoomIn' },
      { role: 'zoomOut' },
      { type: 'separator' },
      { role: 'togglefullscreen' }
    ]
  }
];

const menu = Menu.buildFromTemplate(template);
Menu.setApplicationMenu(menu);
```

### Global Shortcuts

```typescript
// Register global keyboard shortcuts
import { globalShortcut } from 'electron';

app.whenReady().then(() => {
  // Toggle app visibility with ⌘+Shift+Space
  globalShortcut.register('CommandOrControl+Shift+Space', () => {
    const win = BrowserWindow.getFocusedWindow();
    if (win) {
      if (win.isVisible()) {
        win.hide();
      } else {
        win.show();
        win.focus();
      }
    }
  });

  // Open search with ⌘+K (handled in renderer too)
  globalShortcut.register('CommandOrControl+K', () => {
    const win = BrowserWindow.getFocusedWindow();
    win?.webContents.send('trigger-search');
  });
});
```

### System Tray Integration

```typescript
import { Tray, Menu, nativeImage } from 'electron';

let tray: Tray | null = null;

const createTray = () => {
  const icon = nativeImage.createFromPath('assets/tray-icon.png');
  tray = new Tray(icon);

  const contextMenu = Menu.buildFromTemplate([
    { 
      label: 'Show SenterUI', 
      click: () => {
        const win = BrowserWindow.getAllWindows()[0];
        win?.show();
        win?.focus();
      }
    },
    { type: 'separator' },
    { 
      label: 'New Conversation',
      accelerator: 'CommandOrControl+N',
      click: () => {
        // Create new conversation
      }
    },
    { type: 'separator' },
    { 
      label: 'Quit', 
      role: 'quit' 
    }
  ]);

  tray.setContextMenu(contextMenu);
  tray.setToolTip('SenterUI - Local AI Assistant');
};

app.whenReady().then(createTray);
```

### File System Access

```typescript
// Secure file system access through IPC
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Documents', extensions: ['txt', 'pdf', 'doc', 'docx'] },
      { name: 'Code', extensions: ['js', 'ts', 'py', 'java', 'cpp'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });

  if (!result.canceled) {
    return result.filePaths;
  }
  return null;
});

// Read file securely
ipcMain.handle('read-file', async (event, filePath: string) => {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    const stats = await fs.stat(filePath);
    
    return {
      content,
      name: path.basename(filePath),
      size: stats.size,
      type: path.extname(filePath)
    };
  } catch (error) {
    throw new Error(`Failed to read file: ${error.message}`);
  }
});
```

### Screenshot Capture (macOS Native)

```typescript
// Use macOS native screenshot API
ipcMain.handle('capture-screenshot', async () => {
  const { exec } = require('child_process');
  const tempPath = path.join(app.getPath('temp'), `screenshot-${Date.now()}.png`);

  return new Promise((resolve, reject) => {
    // Use macOS screencapture utility
    exec(`screencapture -i ${tempPath}`, (error) => {
      if (error) {
        reject(error);
        return;
      }

      // Read the captured screenshot
      fs.readFile(tempPath, (err, data) => {
        if (err) {
          reject(err);
          return;
        }

        // Clean up temp file
        fs.unlink(tempPath, () => {});

        // Return base64 encoded image
        resolve({
          data: data.toString('base64'),
          mimeType: 'image/png'
        });
      });
    });
  });
});
```

### Permissions Handling

```typescript
// Request microphone permission (macOS)
import { systemPreferences } from 'electron';

const requestMicrophonePermission = async () => {
  const status = systemPreferences.getMediaAccessStatus('microphone');
  
  if (status === 'not-determined') {
    const granted = await systemPreferences.askForMediaAccess('microphone');
    return granted;
  }
  
  return status === 'granted';
};

// Request screen recording permission
const requestScreenPermission = async () => {
  const status = systemPreferences.getMediaAccessStatus('screen');
  
  if (status === 'not-determined' || status === 'denied') {
    // Open System Preferences for user to grant permission
    shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture');
    return false;
  }
  
  return status === 'granted';
};
```

---

## Technical Stack Recommendations

### Core Framework Options

**Option 1: Electron (Recommended)**
```json
{
  "dependencies": {
    "electron": "^28.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.300.0",
    "tailwindcss": "^3.4.0"
  }
}
```

**Pros**: 
- Mature ecosystem
- Extensive documentation
- Better file system access
- Native menu integration

**Option 2: Tauri**
```toml
[dependencies]
tauri = "1.5"
```

**Pros**:
- Smaller bundle size
- Better performance
- Rust-based backend
- Modern architecture

### UI Libraries

**Required**:
- React 18+
- Framer Motion (for animations)
- Lucide React (for icons)
- Tailwind CSS (for styling)

**Recommended**:
```bash
npm install react react-dom
npm install framer-motion
npm install lucide-react
npm install -D tailwindcss postcss autoprefixer
npm install -D @types/react @types/react-dom
npm install -D typescript
```

### Backend/AI Integration

**Local LLM Hosting**:
1. **Ollama** (Recommended for macOS)
   - Easy setup
   - Multiple model support
   - REST API

2. **LM Studio**
   - GUI for model management
   - Compatible API

3. **llama.cpp**
   - Direct C++ integration
   - Maximum performance

**Node.js Client**:
```typescript
// Using Ollama
import ollama from 'ollama';

const response = await ollama.generate({
  model: 'llama2',
  prompt: 'Your prompt here',
  stream: true
});

for await (const chunk of response) {
  console.log(chunk.response);
}
```

### State Management

```typescript
// Using Zustand (lightweight, perfect for this use case)
import create from 'zustand';

interface AppState {
  // Context Menu
  isContextMenuOpen: boolean;
  contextItems: ContextItem[];
  contextFilter: 'all' | 'web' | 'files' | 'clipboard';
  
  // Chat
  isChatViewOpen: boolean;
  chatSessions: ChatSession[];
  activeSessionId: string | null;
  
  // Input
  isMicActive: boolean;
  currentMessage: string;
  
  // Actions
  toggleContextMenu: () => void;
  addContextItem: (item: ContextItem) => void;
  removeContextItem: (id: string) => void;
  setContextFilter: (filter: string) => void;
  
  toggleChatView: () => void;
  addMessage: (sessionId: string, message: ChatMessage) => void;
  createNewSession: () => void;
  
  setMicActive: (active: boolean) => void;
  setCurrentMessage: (message: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  isContextMenuOpen: false,
  contextItems: [],
  contextFilter: 'all',
  isChatViewOpen: false,
  chatSessions: [],
  activeSessionId: null,
  isMicActive: false,
  currentMessage: '',
  
  // Actions
  toggleContextMenu: () => set((state) => ({ 
    isContextMenuOpen: !state.isContextMenuOpen,
    isChatViewOpen: false  // Close chat when opening context
  })),
  
  addContextItem: (item) => set((state) => ({
    contextItems: [item, ...state.contextItems]
  })),
  
  removeContextItem: (id) => set((state) => ({
    contextItems: state.contextItems.filter(item => item.id !== id)
  })),
  
  setContextFilter: (filter) => set({ contextFilter: filter }),
  
  toggleChatView: () => set((state) => ({ 
    isChatViewOpen: !state.isChatViewOpen,
    isContextMenuOpen: false  // Close context when opening chat
  })),
  
  addMessage: (sessionId, message) => set((state) => ({
    chatSessions: state.chatSessions.map(session =>
      session.id === sessionId
        ? { ...session, messages: [...session.messages, message] }
        : session
    )
  })),
  
  createNewSession: () => set((state) => {
    const newSession: ChatSession = {
      id: generateId(),
      title: 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };
    return {
      chatSessions: [newSession, ...state.chatSessions],
      activeSessionId: newSession.id
    };
  }),
  
  setMicActive: (active) => set({ isMicActive: active }),
  setCurrentMessage: (message) => set({ currentMessage: message })
}));
```

### Database (Local Storage)

```typescript
// Using better-sqlite3 for Electron
import Database from 'better-sqlite3';

const db = new Database('senterui.db');

// Initialize tables
db.exec(`
  CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at INTEGER,
    updated_at INTEGER
  );

  CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    timestamp INTEGER,
    metadata TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
  );

  CREATE TABLE IF NOT EXISTS context_items (
    id TEXT PRIMARY KEY,
    type TEXT,
    title TEXT,
    content TEXT,
    preview TEXT,
    timestamp INTEGER,
    metadata TEXT
  );
`);

// CRUD operations
export const dbOperations = {
  // Sessions
  createSession: (session: ChatSession) => {
    const stmt = db.prepare(`
      INSERT INTO sessions (id, title, created_at, updated_at)
      VALUES (?, ?, ?, ?)
    `);
    stmt.run(
      session.id,
      session.title,
      session.createdAt.getTime(),
      session.updatedAt.getTime()
    );
  },

  // Messages
  addMessage: (message: ChatMessage, sessionId: string) => {
    const stmt = db.prepare(`
      INSERT INTO messages (id, session_id, role, content, timestamp, metadata)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      message.id,
      sessionId,
      message.role,
      message.content,
      message.timestamp.getTime(),
      JSON.stringify(message.metadata)
    );
  },

  getSessionMessages: (sessionId: string): ChatMessage[] => {
    const stmt = db.prepare(`
      SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC
    `);
    const rows = stmt.all(sessionId);
    return rows.map(row => ({
      id: row.id,
      role: row.role,
      content: row.content,
      timestamp: new Date(row.timestamp),
      metadata: JSON.parse(row.metadata)
    }));
  }
};
```

---

## Project Structure

```
senterui/
├── src/
│   ├── main/                    # Electron main process
│   │   ├── main.ts             # App entry point
│   │   ├── ipc-handlers.ts     # IPC event handlers
│   │   ├── ai-client.ts        # AI backend integration
│   │   ├── database.ts         # Local database
│   │   └── menu.ts             # Native menu
│   │
│   ├── renderer/               # React frontend
│   │   ├── App.tsx            # Root component
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── InputBar.tsx
│   │   │   ├── ContextMenu.tsx
│   │   │   ├── ChatView.tsx
│   │   │   └── ui/            # Reusable UI components
│   │   │
│   │   ├── hooks/
│   │   │   ├── useAppStore.ts
│   │   │   ├── useAI.ts
│   │   │   └── useKeyboard.ts
│   │   │
│   │   ├── utils/
│   │   │   ├── formatting.ts
│   │   │   └── helpers.ts
│   │   │
│   │   └── styles/
│   │       ├── globals.css
│   │       └── animations.css
│   │
│   ├── preload/               # Electron preload scripts
│   │   └── preload.ts
│   │
│   └── types/                 # TypeScript definitions
│       ├── context.ts
│       ├── chat.ts
│       └── ai.ts
│
├── assets/
│   ├── fonts/
│   │   └── Manrope-VariableFont_wght.ttf
│   ├── icons/
│   └── images/
│
├── public/
│   └── index.html
│
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
└── electron-builder.json      # Build configuration
```

---

## Build & Distribution

### Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Type checking
npm run type-check
```

### Production Build

```bash
# Build for macOS
npm run build:mac

# Create DMG installer
npm run package:mac
```

### electron-builder Configuration

```json
{
  "appId": "com.senterui.app",
  "productName": "SenterUI",
  "directories": {
    "output": "dist"
  },
  "files": [
    "build/**/*",
    "node_modules/**/*",
    "package.json"
  ],
  "mac": {
    "category": "public.app-category.productivity",
    "icon": "assets/icon.icns",
    "hardenedRuntime": true,
    "gatekeeperAssess": false,
    "entitlements": "build/entitlements.mac.plist",
    "entitlementsInherit": "build/entitlements.mac.plist",
    "target": [
      {
        "target": "dmg",
        "arch": ["x64", "arm64"]
      }
    ]
  },
  "dmg": {
    "contents": [
      {
        "x": 130,
        "y": 220
      },
      {
        "x": 410,
        "y": 220,
        "type": "link",
        "path": "/Applications"
      }
    ]
  }
}
```

---

## Testing Checklist

Before considering the implementation complete, verify:

### Visual & Animation
- [ ] All glass panels have correct opacity and blur
- [ ] Ambient lighting effects are visible
- [ ] Rainbow border animation works on mic activation
- [ ] Context menu expands upward smoothly
- [ ] Chat view expands downward smoothly
- [ ] Main sidebar and input bar remain fixed
- [ ] Hover states work on all interactive elements
- [ ] Button animations feel responsive
- [ ] Text is readable on all backgrounds
- [ ] Brand colors are applied consistently

### Functionality
- [ ] Search bar responds to ⌘K shortcut
- [ ] Navigation items change active state
- [ ] Settings dropdown opens/closes
- [ ] Context menu toggles correctly
- [ ] Context items can be added/removed
- [ ] Context filtering works
- [ ] Microphone starts/stops recording
- [ ] Camera captures screenshots
- [ ] Message input expands with text
- [ ] Enter sends message, Shift+Enter creates new line
- [ ] Chat view toggles correctly
- [ ] Messages display with correct styling
- [ ] Chat history loads previous conversations
- [ ] User profile menu opens

### Integration
- [ ] Messages send to AI backend
- [ ] AI responses display in chat
- [ ] Context items are included in queries
- [ ] File selection works
- [ ] Screenshot capture works
- [ ] Voice transcription works (if implemented)
- [ ] Sessions persist between app restarts
- [ ] Multiple chat sessions can be managed

### macOS Specific
- [ ] Native title bar appears correctly
- [ ] Window transparency works
- [ ] Global shortcuts function
- [ ] System tray integration works
- [ ] Menu bar items function
- [ ] Permissions are requested properly
- [ ] App icon displays correctly
- [ ] DMG installer creates properly

---

## Conclusion

This specification provides a complete blueprint for rebuilding the SenterUI application as a local macOS AI assistant. The key architectural principles are:

1. **Fixed Core UI**: The sidebar and input bar never move
2. **Expanding Panels**: Context menu (up) and chat view (down) animate from input bar
3. **Liquid Glass Aesthetic**: Consistent translucent design with backdrop blur
4. **Brand Identity**: Strict adherence to color palette and Manrope typography
5. **Local AI Integration**: Designed to work with local LLM inference engines
6. **Native macOS Experience**: Leverages platform features and conventions

The UI is designed to be visually stunning while remaining highly functional, providing a seamless interface for interacting with local AI agents. All animations, interactions, and visual effects have been specified in detail to ensure the implementation matches the design vision.

For any questions or clarifications during implementation, refer back to the relevant sections of this specification.
