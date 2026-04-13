# Karakalpak POS + Morphological Analyzer - Frontend

A modern, user-friendly React.js frontend for the Karakalpak POS tagging and morphological analysis API.

## Features

- 🎨 **Beautiful UI**: Modern gradient design with responsive layout
- ⚡ **Real-time Analysis**: Instant POS tagging and morphological analysis
- 📊 **Visual Results**: Clear token cards with detailed linguistic information
- 🔧 **Configurable**: Custom API URL and API key support
- 📱 **Mobile Responsive**: Works on all device sizes
- 🌐 **CORS Ready**: Configured proxy for local development

## Quick Start

### Development Mode

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000` with automatic API proxying to `http://localhost:8000`.

### Production Build

```bash
cd frontend
npm run build
```

Built files are output to `frontend/dist/` and automatically included in the Docker image.

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx          # Main application component
│   ├── main.jsx         # React entry point
│   └── index.css        # Global styles
├── public/              # Static assets
├── dist/                # Production build output
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite configuration
└── index.html           # HTML template
```

## Integration with Backend

The frontend is designed to work seamlessly with the FastAPI backend:

1. **Multi-stage Docker Build**: The frontend is built during Docker image creation
2. **Static File Serving**: Backend serves the built frontend from `/static` directory
3. **API Endpoints**: Uses `/api/analyze` endpoint for text analysis
4. **Error Handling**: Graceful error messages for API failures

## Usage

1. Enter Karakalpak text in the input field
2. Click "Analyze Text" or load an example
3. View results with:
   - Token count and processing time
   - Individual token cards showing:
     - Original text
     - Part of Speech (POS) tag
     - Lemma (base form)
     - Morphological features
     - Confidence score

## Configuration

### Environment Variables

- `VITE_API_URL`: Backend API URL (default: uses relative path)
- Set via API Settings panel in the UI

### API Key

If your backend requires authentication, enter your API key in the settings panel.

## Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

### Adding New Features

1. Create new components in `src/components/`
2. Update styles in `src/index.css` or create component-specific CSS
3. Modify API calls in `App.jsx` as needed

## Deployment

The frontend is automatically deployed with the backend via:

1. **GitHub Actions CI/CD**: Builds frontend on every push to main
2. **Docker Multi-stage Build**: Includes built frontend in final image
3. **Single Container**: No separate frontend hosting needed

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## License

MIT License - See project root for details
