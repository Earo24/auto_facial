# AutoFacial Web Application

A modern web interface for the Film/TV Face Recognition Automation System.

## Features

- **Dashboard**: System overview and quick actions
- **Video Processing**: Upload and process videos with real-time progress
- **Clustering Annotation**: Review auto-clustered faces and label characters
- **Recognition Results**: View and verify batch recognition results
- **Analysis**: Character statistics with interactive charts
- **Settings**: Configure detection and clustering parameters

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development
- **Tailwind CSS** for styling
- **Recharts** for data visualization
- **Lucide React** for icons

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd web_app
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
```

### Preview

```bash
npm run preview
```

## Design System

The application uses a professional dark mode theme inspired by video editing software:

- **Primary Color**: Blue (#3b82f6)
- **Accent Color**: Orange (#f97316)
- **Background**: Dark grays (#0a0a0a to #808080)
- **Typography**: Fira Sans (body), Fira Code (mono)

## Project Structure

```
web_app/
├── src/
│   ├── components/
│   │   ├── layout/       # Layout components (Sidebar, Header)
│   │   └── ui/           # Reusable UI components
│   ├── pages/            # Page components
│   ├── lib/              # Utility functions
│   ├── App.tsx           # Main app component
│   └── main.tsx          # Entry point
├── public/               # Static assets
├── index.html            # HTML template
├── package.json          # Dependencies
├── tailwind.config.js    # Tailwind configuration
└── vite.config.ts        # Vite configuration
```

## Components

### UI Components

- `Button` - Primary, accent, ghost, outline variants
- `Card` / `CardLG` - Card containers
- `Input` - Text input fields
- `Select` - Dropdown select
- `Badge` - Status badges
- `Progress` - Progress bars

### Layout

- `Sidebar` - Collapsible navigation sidebar
- `Header` - Top header with search
- `Layout` - Main layout wrapper

## Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Dashboard | System overview and stats |
| `/processing` | Video Processing | Upload and process videos |
| `/clustering` | Clustering | Face clustering annotation |
| `/recognition` | Recognition | View recognition results |
| `/analysis` | Analysis | Character statistics |
| `/settings` | Settings | Configuration |

## Color Palette

| Role | Hex | Usage |
|------|-----|-------|
| Primary | #3b82f6 | Links, active states |
| Accent | #f97316 | CTAs, important actions |
| Background | #0a0a0a | Main background |
| Card | #1a1a1a | Card backgrounds |
| Border | #404040 | Borders and dividers |

## License

MIT
