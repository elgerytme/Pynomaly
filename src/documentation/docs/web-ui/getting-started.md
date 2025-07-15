# Getting Started with Pynomaly Web UI

This guide will help you get up and running with the Pynomaly Web UI in just a few minutes.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or higher**
- **pip** (Python package installer)
- **Web browser** (Chrome, Firefox, Safari, or Edge)

### Optional Requirements

- **Docker** (for containerized deployment)
- **PostgreSQL** (for production database)
- **Redis** (for enhanced performance)

## 🚀 Installation

### Option 1: Quick Install (Recommended)

Install Pynomaly with web UI support:

```bash
pip install pynomaly[web]
```

### Option 2: Development Install

For development or advanced features:

```bash
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly
pip install -e .[web,dev]
```

### Option 3: Docker Install

Run with Docker:

```bash
docker pull pynomaly/pynomaly:latest
docker run -p 8000:8000 pynomaly/pynomaly:latest
```

## 🏁 First Launch

### 1. Start the Web Interface

```bash
pynomaly web start
```

You should see output similar to:

```
🚀 Starting Pynomaly Web UI...
📊 Dashboard available at: http://localhost:8000
🔧 API documentation at: http://localhost:8000/docs
✅ Server started successfully!
```

### 2. Access the Interface

Open your web browser and navigate to:

```
http://localhost:8000
```

### 3. Initial Setup

On first launch, you'll see the **Welcome Screen** with:

- System status check
- Quick setup wizard
- Sample data option

## 📊 Your First Anomaly Detection

Let's walk through creating your first anomaly detection workflow:

### Step 1: Upload Sample Data

1. **Click "Datasets"** in the navigation menu
2. **Click "Upload Dataset"** button
3. **Choose "Sample Data"** or upload your own CSV file
4. **Fill in dataset details:**
   - Name: "My First Dataset"
   - Description: "Sample data for testing"
5. **Click "Upload"**

### Step 2: Create a Detector

1. **Click "Detectors"** in the navigation menu
2. **Click "Create Detector"** button
3. **Configure the detector:**
   - Name: "My First Detector"
   - Algorithm: "Isolation Forest"
   - Contamination: 0.1 (10% anomalies expected)
4. **Click "Create"**

### Step 3: Train the Detector

1. **Click on your detector** to open details
2. **Click "Train"** button
3. **Select your dataset** from the dropdown
4. **Click "Start Training"**
5. **Wait for training to complete** (usually 10-30 seconds)

### Step 4: Run Detection

1. **Navigate to "Detection"** page
2. **Select your trained detector**
3. **Select your dataset**
4. **Click "Run Detection"**
5. **View the results** in real-time

### Step 5: Analyze Results

1. **Check the detection summary:**
   - Number of anomalies found
   - Anomaly rate percentage
   - Confidence scores
2. **View visualizations:**
   - Scatter plots of data points
   - Anomaly score distributions
   - Time series analysis (if applicable)

## 🎯 Next Steps

### Explore Key Features

1. **Dashboard Overview**
   - System health monitoring
   - Recent activity feed
   - Quick action buttons

2. **Advanced Detection**
   - Ensemble methods
   - Custom parameters
   - Batch processing

3. **Data Analysis**
   - Interactive visualizations
   - Export capabilities
   - Statistical summaries

4. **Monitoring & Alerts**
   - Performance metrics
   - Error tracking
   - Real-time notifications

### Learn More

- **[UI Overview](./ui-overview.md)** - Detailed interface walkthrough
- **[Features Guide](./features.md)** - Complete feature documentation
- **[Configuration](./configuration.md)** - Customize your setup

## ⚙️ Configuration

### Basic Configuration

Create a configuration file at `~/.pynomaly/config.yaml`:

```yaml
# Basic settings
debug: false
log_level: INFO

# Database
database:
  url: sqlite:///pynomaly.db
  
# Web UI
web_ui:
  host: localhost
  port: 8000
  workers: 4

# Security
security:
  secret_key: your-secret-key-here
  session_timeout: 3600
```

### Environment Variables

You can also configure using environment variables:

```bash
export PYNOMALY_DATABASE_URL="postgresql://user:pass@localhost/pynomaly"
export PYNOMALY_SECRET_KEY="your-secret-key"
export PYNOMALY_LOG_LEVEL="INFO"
```

## 🔧 Command Line Interface

### Start the Web UI

```bash
pynomaly web start                    # Start with default settings
pynomaly web start --port 9000        # Start on custom port
pynomaly web start --host 0.0.0.0     # Allow external connections
pynomaly web start --workers 8        # Use 8 worker processes
```

### Development Mode

```bash
pynomaly web dev                      # Start in development mode
pynomaly web dev --reload             # Auto-reload on file changes
```

### Database Management

```bash
pynomaly db init                      # Initialize database
pynomaly db migrate                   # Run migrations
pynomaly db reset                     # Reset database (careful!)
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  pynomaly:
    image: pynomaly/pynomaly:latest
    ports:
      - "8000:8000"
    environment:
      - PYNOMALY_DATABASE_URL=postgresql://postgres:password@db:5432/pynomaly
      - PYNOMALY_REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
      
  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=pynomaly
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    
volumes:
  postgres_data:
```

Start with:

```bash
docker-compose up -d
```

## 🆘 Troubleshooting

### Common Issues

**Port already in use:**

```bash
pynomaly web start --port 8001
```

**Permission denied:**

```bash
sudo pynomaly web start --port 80
```

**Database connection error:**

- Check database configuration
- Ensure database is running
- Verify connection string

**Module not found:**

```bash
pip install pynomaly[web] --upgrade
```

### Getting Help

1. **Check the logs:**

   ```bash
   pynomaly logs --tail 100
   ```

2. **Verify installation:**

   ```bash
   pynomaly --version
   pynomaly web check
   ```

3. **Reset configuration:**

   ```bash
   pynomaly config reset
   ```

### Performance Tips

- Use PostgreSQL for production databases
- Enable Redis for caching
- Increase worker processes for high load
- Configure reverse proxy (Nginx) for production

## ✅ Verification Checklist

Before proceeding, ensure:

- [ ] Pynomaly Web UI starts without errors
- [ ] Dashboard loads in your browser
- [ ] You can create and upload a dataset
- [ ] You can create and train a detector
- [ ] Detection runs successfully
- [ ] Results are displayed correctly

## 🎉 Success

Congratulations! You've successfully set up the Pynomaly Web UI. You're now ready to:

- Explore advanced features
- Customize your configuration
- Scale for production use
- Integrate with your existing systems

**Next:** Continue to the [UI Overview](./ui-overview.md) to learn about the interface in detail.
