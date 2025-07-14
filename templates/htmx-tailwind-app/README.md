# HTMX + Tailwind CSS App Template

A modern web application template combining HTMX for dynamic interactions with Tailwind CSS for styling, built on FastAPI.

## Features

- **HTMX**: Dynamic web apps with minimal JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **FastAPI**: High-performance Python web framework
- **Jinja2**: Template engine for server-side rendering
- **Hot Reload**: Development server with auto-reload
- **Component System**: Reusable UI components
- **Form Handling**: HTMX-powered forms with validation
- **Real-time Updates**: WebSockets and Server-Sent Events
- **Progressive Enhancement**: Works without JavaScript
- **Responsive Design**: Mobile-first approach

## Directory Structure

```
htmx-tailwind-app/
├── build/                 # Build artifacts
├── deploy/                # Deployment configurations
├── docs/                  # Documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary files
├── src/                   # Source code
│   └── htmx_app/
│       ├── api/          # API endpoints
│       ├── templates/    # Jinja2 templates
│       ├── static/       # Static assets
│       ├── components/   # Reusable components
│       ├── core/         # Core logic
│       ├── models/       # Data models
│       └── utils/        # Utilities
├── pkg/                  # Package metadata
├── examples/             # Usage examples
├── tests/                # Test suites
├── .github/              # GitHub workflows
├── scripts/              # Build scripts
├── tailwind.config.js    # Tailwind configuration
├── package.json          # Node.js dependencies
├── pyproject.toml        # Python configuration
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Local development
├── README.md            # Project documentation
├── TODO.md              # Task tracking
└── CHANGELOG.md         # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> htmx-tailwind-app
   cd htmx-tailwind-app
   ```

2. **Install dependencies**:
   ```bash
   # Python dependencies
   pip install -e .
   
   # Node.js dependencies for Tailwind
   npm install
   ```

3. **Build CSS**:
   ```bash
   npm run build-css
   ```

4. **Start development server**:
   ```bash
   # Terminal 1: Python server
   uvicorn htmx_app.main:app --reload
   
   # Terminal 2: Tailwind watcher
   npm run watch-css
   ```

5. **Open the app**:
   - App: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Development

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -e ".[dev,test]"

# Install Node.js dependencies
npm install

# Build initial CSS
npm run build-css
```

### Running in Development

```bash
# Start Python server with hot reload
uvicorn htmx_app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, watch CSS changes
npm run watch-css
```

### Building for Production

```bash
# Build optimized CSS
npm run build-css:prod

# Run production server
gunicorn htmx_app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## HTMX Integration

### Basic HTMX Usage

```html
<!-- Load more content -->
<button hx-get="/api/load-more" 
        hx-target="#content" 
        hx-swap="afterend">
    Load More
</button>

<!-- Form submission -->
<form hx-post="/api/submit" 
      hx-target="#result">
    <input type="text" name="message" required>
    <button type="submit">Submit</button>
</form>

<!-- Real-time updates -->
<div hx-get="/api/status" 
     hx-trigger="every 2s"
     hx-swap="innerHTML">
    Loading...
</div>
```

### Advanced HTMX Features

```html
<!-- Infinite scroll -->
<div hx-get="/api/items" 
     hx-trigger="revealed"
     hx-swap="afterend">
    <div class="loading">Loading more items...</div>
</div>

<!-- Conditional requests -->
<div hx-get="/api/data" 
     hx-trigger="load"
     hx-headers='{"X-Requested-With": "XMLHttpRequest"}'>
</div>

<!-- WebSocket integration -->
<div hx-ws="connect:/ws/updates"
     hx-swap="innerHTML">
    <div id="messages"></div>
</div>
```

## Tailwind CSS

### Configuration

The `tailwind.config.js` includes:
- Custom color palette
- Extended spacing scale
- Custom components
- Responsive breakpoints
- Dark mode support

### Utility Classes

```html
<!-- Layout -->
<div class="container mx-auto px-4 py-8">
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <!-- Grid items -->
    </div>
</div>

<!-- Components -->
<button class="btn btn-primary">
    Primary Button
</button>

<div class="card">
    <div class="card-header">
        <h3 class="card-title">Card Title</h3>
    </div>
    <div class="card-body">
        Card content
    </div>
</div>
```

### Custom Components

```css
/* Custom button component */
.btn {
    @apply px-4 py-2 rounded-md font-medium transition-colors;
}

.btn-primary {
    @apply bg-blue-600 text-white hover:bg-blue-700;
}

.btn-secondary {
    @apply bg-gray-200 text-gray-900 hover:bg-gray-300;
}

/* Card component */
.card {
    @apply bg-white rounded-lg shadow-md overflow-hidden;
}

.card-header {
    @apply px-6 py-4 bg-gray-50 border-b;
}

.card-body {
    @apply px-6 py-4;
}
```

## API Endpoints

### Pages
- `GET /` - Home page
- `GET /dashboard` - Dashboard page
- `GET /profile` - User profile page

### HTMX Endpoints
- `GET /api/components/{component}` - Load component
- `POST /api/forms/{form}` - Handle form submission
- `GET /api/data/{endpoint}` - Fetch data
- `GET /api/search` - Search results
- `POST /api/actions/{action}` - Execute action

### WebSocket
- `WS /ws/updates` - Real-time updates
- `WS /ws/chat` - Chat functionality

## Components

### Reusable Components

```python
# Component renderer
@router.get("/api/components/user-card")
async def user_card(user_id: int):
    user = get_user(user_id)
    return templates.TemplateResponse(
        "components/user_card.html",
        {"request": request, "user": user}
    )
```

```html
<!-- components/user_card.html -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">{{ user.name }}</h3>
    </div>
    <div class="card-body">
        <p class="text-gray-600">{{ user.email }}</p>
        <button hx-post="/api/actions/follow" 
                hx-vals='{"user_id": {{ user.id }}}' 
                class="btn btn-primary mt-2">
            Follow
        </button>
    </div>
</div>
```

## Form Handling

### HTMX Forms

```html
<!-- Contact form -->
<form hx-post="/api/forms/contact"
      hx-target="#form-result"
      hx-swap="innerHTML"
      class="space-y-4">
    
    <div>
        <label class="block text-sm font-medium text-gray-700">
            Name
        </label>
        <input type="text" 
               name="name" 
               required
               class="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
    </div>
    
    <div>
        <label class="block text-sm font-medium text-gray-700">
            Email
        </label>
        <input type="email" 
               name="email" 
               required
               class="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
    </div>
    
    <div>
        <label class="block text-sm font-medium text-gray-700">
            Message
        </label>
        <textarea name="message" 
                  rows="4"
                  required
                  class="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
        </textarea>
    </div>
    
    <button type="submit" 
            class="btn btn-primary w-full">
        Send Message
    </button>
</form>

<div id="form-result" class="mt-4"></div>
```

### Form Validation

```python
@router.post("/api/forms/contact")
async def handle_contact_form(request: Request, form: ContactForm):
    try:
        # Validate form data
        if not form.name or not form.email or not form.message:
            return templates.TemplateResponse(
                "components/form_error.html",
                {"request": request, "error": "All fields are required"}
            )
        
        # Process form
        await send_contact_email(form)
        
        return templates.TemplateResponse(
            "components/form_success.html",
            {"request": request, "message": "Message sent successfully!"}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {"request": request, "error": str(e)}
        )
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/htmx_app

# Run specific test file
pytest tests/test_components.py

# Run integration tests
pytest tests/integration/
```

## Deployment

### Docker

```bash
# Build image
docker build -t htmx-tailwind-app:latest .

# Run container
docker run -p 8000:8000 htmx-tailwind-app:latest

# Use docker-compose
docker-compose up -d
```

### Production Checklist

- [ ] Build optimized CSS with `npm run build-css:prod`
- [ ] Set environment variables
- [ ] Configure reverse proxy (nginx)
- [ ] Set up SSL certificates
- [ ] Configure monitoring
- [ ] Set up logging
- [ ] Configure backup strategy

## Scripts

- `npm run build-css` - Build CSS for development
- `npm run build-css:prod` - Build optimized CSS for production
- `npm run watch-css` - Watch CSS changes during development
- `npm run lint-css` - Lint CSS files
- `python -m pytest` - Run Python tests
- `npm run build` - Build complete application

## License

MIT License - see LICENSE file for details