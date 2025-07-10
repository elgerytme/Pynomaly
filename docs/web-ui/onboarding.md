# User Onboarding Guide

Welcome to Pynomaly! This interactive onboarding guide will help you get familiar with the platform and create your first successful anomaly detection workflow.

## 🎯 Learning Objectives

By the end of this guide, you will:

- Understand the Pynomaly interface and navigation
- Successfully upload and analyze your first dataset
- Create and train your first anomaly detector
- Run anomaly detection and interpret results
- Know where to find help and advanced features

## 📚 Before We Start

### What You'll Need

- **Sample Data**: We'll provide sample datasets, or you can use your own CSV file
- **10-15 Minutes**: This guide takes about 10-15 minutes to complete
- **Web Browser**: Modern browser (Chrome, Firefox, Safari, or Edge)

### Quick Tips

- 💡 Look for these tip boxes throughout the guide
- ⚠️ Pay attention to warning boxes for important information
- 📋 Check off tasks as you complete them

## 🚀 Step 1: Interface Familiarization

### Main Navigation Tour

Let's start by exploring the main interface:

**📋 Task 1.1: Explore the Dashboard**

- [ ] Click on **Dashboard** (if not already there)
- [ ] Notice the key metrics cards at the top
- [ ] Scroll down to see the activity feed
- [ ] Try clicking on different metric cards

**📋 Task 1.2: Navigation Menu**

- [ ] Hover over each menu item to see what they do:
  - **Detectors**: Manage your anomaly detection algorithms
  - **Datasets**: Upload and manage your data
  - **Detection**: Run anomaly detection
  - **Experiments**: Track and compare experiments
  - **Visualizations**: Advanced charts and analysis

💡 **Tip**: The breadcrumb navigation at the top shows where you are and helps you navigate back.

### Quick Actions

- [ ] Notice the **Quick Actions** panel on the right
- [ ] These provide shortcuts to common tasks

## 📊 Step 2: Your First Dataset

Let's upload and explore a dataset:

**📋 Task 2.1: Navigate to Datasets**

- [ ] Click on **Datasets** in the main navigation
- [ ] You should see the datasets page with an upload button

**📋 Task 2.2: Upload Sample Data**

Option A - Use Sample Data:

- [ ] Click **"Use Sample Data"** if available
- [ ] Select **"Credit Card Fraud"** or **"Network Intrusion"** dataset
- [ ] Click **"Load Sample"**

Option B - Upload Your Own Data:

- [ ] Click **"Upload Dataset"**
- [ ] Choose a CSV file (preferably with numerical data)
- [ ] Fill in the form:
  - **Name**: Give it a descriptive name
  - **Description**: Brief description of your data
- [ ] Click **"Upload"**

**📋 Task 2.3: Explore Your Dataset**

- [ ] Click on your dataset name to view details
- [ ] Explore the different tabs:
  - **Overview**: Basic statistics and information
  - **Data Quality**: Quality assessment and recommendations
  - **Preview**: Sample rows from your dataset
  - **Visualizations**: Basic charts and distributions

💡 **Tip**: The Data Quality tab provides valuable insights about missing values, outliers, and data types.

**📋 Task 2.4: Understand Data Quality**

- [ ] Review the quality score (aim for 80% or higher)
- [ ] Check for missing values and outliers
- [ ] Note any recommendations for data preprocessing

⚠️ **Important**: Good data quality is crucial for effective anomaly detection. Address any major issues before proceeding.

## 🔍 Step 3: Create Your First Detector

Now let's create an anomaly detector:

**📋 Task 3.1: Navigate to Detectors**

- [ ] Click on **Detectors** in the main navigation
- [ ] Click the **"Create Detector"** button

**📋 Task 3.2: Configure Your Detector**

- [ ] Fill in the basic information:
  - **Name**: "My First Detector"
  - **Description**: "Learning how to detect anomalies"
  - **Algorithm**: Select "Isolation Forest" (recommended for beginners)

**📋 Task 3.3: Set Parameters**

- [ ] **Contamination**: Set to 0.1 (expects 10% anomalies)
- [ ] **Random State**: Set to 42 (for reproducible results)
- [ ] Leave other parameters as default for now

💡 **Tip**: Isolation Forest is great for beginners because it works well without much tuning and handles different data types effectively.

**📋 Task 3.4: Create the Detector**

- [ ] Review your settings
- [ ] Click **"Create Detector"**
- [ ] You should see your new detector in the list

## 🎯 Step 4: Train Your Detector

Time to train your detector on the data:

**📋 Task 4.1: Start Training**

- [ ] Click on your detector name to open details
- [ ] Click the **"Train"** button
- [ ] Select your dataset from the dropdown
- [ ] Click **"Start Training"**

**📋 Task 4.2: Monitor Training Progress**

- [ ] Watch the progress bar fill up
- [ ] Training should complete in 10-30 seconds for sample data
- [ ] Check for any error messages

**📋 Task 4.3: Review Training Results**

- [ ] Once complete, review the training summary:
  - **Training Time**: How long it took
  - **Model Size**: Memory usage of the trained model
  - **Feature Importance**: Which features are most important

💡 **Tip**: If training fails, check your data quality and try adjusting the contamination parameter.

## 🚨 Step 5: Run Anomaly Detection

Now for the exciting part - finding anomalies!

**📋 Task 5.1: Navigate to Detection**

- [ ] Click on **Detection** in the main navigation
- [ ] This is where you run anomaly detection

**📋 Task 5.2: Configure Detection**

- [ ] **Select Detector**: Choose your trained detector
- [ ] **Select Dataset**: Choose your dataset (can be the same one used for training)
- [ ] **Output Options**: Leave as default for now

**📋 Task 5.3: Run Detection**

- [ ] Click **"Run Detection"**
- [ ] Watch the real-time progress
- [ ] Detection should complete in a few seconds

**📋 Task 5.4: Examine Results**

- [ ] Review the detection summary:
  - **Total Samples**: Number of data points analyzed
  - **Anomalies Found**: Number of anomalies detected
  - **Anomaly Rate**: Percentage of data points flagged as anomalous
  - **Processing Time**: How long detection took

## 📈 Step 6: Analyze and Interpret Results

Let's understand what the results mean:

**📋 Task 6.1: View Result Details**

- [ ] Click **"View Detailed Results"**
- [ ] Explore the results table showing:
  - Each data point with its anomaly score
  - Whether it was classified as normal or anomalous
  - Confidence levels

**📋 Task 6.2: Explore Visualizations**

- [ ] Look at the scatter plot showing normal vs. anomalous points
- [ ] Check the anomaly score distribution histogram
- [ ] If your data has timestamps, explore the time series view

**📋 Task 6.3: Understand Anomaly Scores**

- [ ] Higher scores = more anomalous
- [ ] Scores typically range from 0 to 1
- [ ] Points above the threshold are classified as anomalies

💡 **Tip**: Click on individual anomalous points to see which features contributed most to the anomaly score.

**📋 Task 6.4: Export Results (Optional)**

- [ ] Try exporting results in different formats:
  - CSV for spreadsheet analysis
  - JSON for programmatic access
  - PDF for reporting

## 🎓 Step 7: Explore Advanced Features

Now that you've mastered the basics, let's explore some advanced features:

**📋 Task 7.1: Experiment Tracking**

- [ ] Navigate to **Experiments**
- [ ] See how your detection run was automatically tracked
- [ ] Compare different runs and parameters

**📋 Task 7.2: Advanced Visualizations**

- [ ] Go to **Visualizations**
- [ ] Try creating custom charts with your data
- [ ] Experiment with different visualization types

**📋 Task 7.3: Ensemble Methods (If Available)**

- [ ] Navigate to **Ensemble**
- [ ] Learn about combining multiple detectors
- [ ] Try creating a simple ensemble

**📋 Task 7.4: Monitoring Dashboard**

- [ ] Check out **Monitoring**
- [ ] See system health and performance metrics
- [ ] Understand resource usage

## 🔄 Step 8: Next Steps and Best Practices

Congratulations! You've completed your first anomaly detection workflow. Here's what to do next:

### Immediate Next Steps

**📋 Task 8.1: Try Different Algorithms**

- [ ] Create another detector with a different algorithm (e.g., Local Outlier Factor)
- [ ] Compare results between different algorithms
- [ ] Notice how different algorithms may find different anomalies

**📋 Task 8.2: Experiment with Parameters**

- [ ] Try different contamination levels (0.05, 0.15, 0.2)
- [ ] See how this affects the number of anomalies found
- [ ] Find the sweet spot for your data

**📋 Task 8.3: Upload Your Own Data**

- [ ] If you used sample data, try uploading your own dataset
- [ ] Apply what you learned to real data from your domain

### Best Practices You've Learned

✅ **Data Quality First**: Always check data quality before training
✅ **Start Simple**: Begin with well-established algorithms like Isolation Forest
✅ **Monitor Training**: Watch for errors and warnings during training
✅ **Interpret Results**: Don't just look at numbers - understand what they mean
✅ **Iterate and Improve**: Experiment with different parameters and approaches

### Advanced Topics to Explore

🔮 **Coming Next**: Consider exploring these topics as you become more comfortable:

1. **Feature Engineering**: Modify your data to improve detection
2. **Ensemble Methods**: Combine multiple detectors for better results
3. **AutoML**: Let the system automatically find the best parameters
4. **Explainability**: Understand why specific points are anomalous
5. **Real-time Detection**: Set up continuous anomaly monitoring

## 📚 Getting Help and Learning More

### Built-in Help

- [ ] **Tooltips**: Hover over ? icons for context-sensitive help
- [ ] **Help Panel**: Press F1 or click the help icon for detailed explanations
- [ ] **Interactive Tours**: Access guided tours from the help menu

### Documentation and Resources

- [ ] **User Guide**: Comprehensive documentation (you're reading it!)
- [ ] **API Documentation**: For programmatic access
- [ ] **Video Tutorials**: Visual learning resources
- [ ] **Example Gallery**: Real-world use cases and examples

### Community and Support

- [ ] **Community Forum**: Ask questions and share experiences
- [ ] **Knowledge Base**: Searchable help articles
- [ ] **Bug Reports**: Report issues and suggest improvements

## 🎉 Congratulations

You've successfully completed the Pynomaly onboarding experience! You now know how to:

✅ Navigate the Pynomaly interface  
✅ Upload and analyze datasets  
✅ Create and train anomaly detectors  
✅ Run anomaly detection  
✅ Interpret and analyze results  
✅ Access help and documentation  

### What's Next?

1. **Practice**: Try the workflow with different datasets and algorithms
2. **Explore**: Dive deeper into advanced features that interest you
3. **Apply**: Use Pynomaly for real anomaly detection challenges in your work
4. **Share**: Help others by sharing your experiences and learnings

### Quick Reference Card

Keep this handy for future reference:

**Basic Workflow**:
Datasets → Upload → Detectors → Create → Train → Detection → Run → Analyze

**Key Shortcuts**:

- `Ctrl+H`: Toggle help panel
- `Ctrl+/`: Search documentation
- `F1`: Context-sensitive help

**Need Help?**:

- Look for ? icons for tooltips
- Check the help panel (F1)
- Visit the community forum
- Contact support for complex issues

---

**Ready to become a Pynomaly expert?** Continue exploring the interface and don't hesitate to experiment. The best way to learn is by doing!

**Feedback**: Help us improve this onboarding experience by sharing your feedback through the help menu.
