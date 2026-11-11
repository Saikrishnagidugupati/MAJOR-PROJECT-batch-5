# TicketSense-XL

### A Hybrid NLP and Multi-Classifier Framework for Scalable Customer Support Ticket Analysis and Decision Automation

TicketSense-XL is an AI-powered customer support ticket analysis system
that combines Natural Language Processing (NLP), XLNet contextual
embeddings, and classical machine-learning classifiers to support
automated ticket triage and decision-making.

The system analyzes customer support ticket text and predicts three
important attributes:

-   **Ticket Priority** -- estimates the urgency of a customer issue.
-   **Customer Satisfaction Rating** -- predicts the satisfaction
    category represented in the available dataset.
-   **Resolution** -- predicts the resolution category or status
    represented by the available labels.

It is designed to reduce repetitive manual ticket classification,
support faster triage, and provide useful insights to customer-support
teams.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Objectives](#objectives)
-   [Key Features](#key-features)
-   [System Workflow](#system-workflow)
-   [Machine Learning Models](#machine-learning-models)
-   [Project Modules](#project-modules)
-   [Evaluation](#evaluation)
-   [Technology](#technology)
-   [Repository Structure](#repository-structure)
-   [Getting Started](#getting-started)
-   [Using the Application](#using-the-application)
-   [Future Enhancements](#future-enhancements)
-   [Project Team](#project-team)
-   [Acknowledgements](#acknowledgements)

------------------------------------------------------------------------

## Project Overview

Traditional customer-support systems often depend on agents to read,
categorize, and prioritize incoming tickets. When ticket volumes
increase, manual processing can lead to delays, inconsistent
classification, and additional workload.

TicketSense-XL addresses this challenge through a hybrid approach:

1.  Customer ticket data is ingested and validated.
2.  Ticket text is cleaned and prepared for NLP processing.
3.  XLNet converts the text into contextual numerical feature
    representations.
4.  Candidate machine-learning classifiers analyze the extracted
    features.
5.  The system predicts ticket priority, customer satisfaction, and
    resolution.
6.  Similar-ticket analysis and knowledge-base information can support
    agent decisions.
7.  Results and model performance can be presented through an analytics
    interface.

The architecture is modular, allowing individual components to be
tested, maintained, and extended.

## Objectives

-   Build a preprocessing pipeline for customer-support ticket text.
-   Extract contextual features using the XLNet transformer model.
-   Train and compare multiple machine-learning classifiers.
-   Predict ticket priority, customer satisfaction rating, and
    resolution.
-   Evaluate models using standard classification metrics.
-   Support ticket analysis through visualizations and reports.
-   Provide a foundation for similarity detection and knowledge-assisted
    recommendations.

## Key Features

-   **Text preprocessing:** normalization, tokenization, and preparation
    of ticket descriptions.
-   **Contextual feature extraction:** XLNet-based embeddings to
    represent ticket meaning.
-   **Multi-target prediction:** prediction of priority, satisfaction,
    and resolution attributes.
-   **Classifier comparison:** evaluation of several candidate
    machine-learning algorithms.
-   **Ticket analysis:** support for identifying related or potentially
    duplicate tickets.
-   **Decision support:** potential use of historical tickets and
    knowledge-base guidance.
-   **Analytics:** ticket distributions, prediction summaries, and model
    evaluation reports.
-   **Human review:** uncertain or high-impact recommendations can be
    referred to support agents.

> Similarity results indicate related text or meaning; they do not
> independently confirm that two tickets have the same root cause.
> Recommendations should be reviewed by an authorized support agent.

## System Workflow

``` text
Customer Ticket Input
        |
        v
Data Ingestion and Validation
        |
        v
NLP Text Preprocessing
        |
        v
XLNet Tokenization and Embedding Extraction
        |
        v
Hybrid Machine-Learning Classifiers
        |
        v
Multi-Target Prediction
(Priority | Satisfaction | Resolution)
        |
        v
Similarity Detection and Knowledge Assistance
        |
        v
Support-Agent Review and Decision Support
        |
        v
Analytics and Reports
```

## Machine Learning Models

The project report describes the following candidate classifiers:

  -----------------------------------------------------------------------
  Model                               Description
  ----------------------------------- -----------------------------------
  **QDA**                             Quadratic Discriminant Analysis;
                                      models class-specific covariance
                                      and quadratic decision boundaries.

  **LDA**                             Linear Discriminant Analysis; uses
                                      a shared covariance assumption and
                                      linear decision boundaries.

  **HGB**                             Histogram-based Gradient Boosting;
                                      builds boosted decision trees using
                                      histogram-based splitting.

  **SGD**                             Stochastic Gradient Descent
                                      Classifier; trains linear models
                                      through incremental parameter
                                      updates.

  **NC**                              Nearest Centroid; assigns a class
                                      based on the closest class
                                      centroid.
  -----------------------------------------------------------------------

XLNet generates contextual feature vectors that are supplied to the
candidate classifiers. The report also describes using HGB classifiers
for the three target predictions, with a separate model trained for each
target.

## Project Modules

1.  **Customer Support Ticket Input** -- collects ticket subject,
    description, product details, ticket type, and metadata.
2.  **Data Ingestion and Validation** -- checks required fields, missing
    values, formats, and duplicate records where applicable.
3.  **NLP Text Preprocessing** -- cleans and normalizes text while
    preparing it for XLNet tokenization.
4.  **XLNet Feature Extraction** -- generates contextual embeddings and
    fixed-length feature vectors.
5.  **Hybrid Multi-Classifier** -- trains and compares QDA, LDA, HGB,
    SGD, and NC.
6.  **Multi-Target Prediction** -- predicts priority, satisfaction, and
    resolution labels.
7.  **Similarity and Duplicate Ticket Detection** -- retrieves
    semantically related historical tickets.
8.  **Knowledge-Base Integration** -- provides access to previously
    resolved cases, troubleshooting procedures, and approved guidance.
9.  **Decision Automation and Recommendation** -- combines model outputs
    and related information to assist support agents.
10. **Model Evaluation** -- compares classifiers using accuracy,
    precision, recall, and F1-score.
11. **Analytics and Visualization** -- presents ticket patterns,
    prediction summaries, and evaluation metrics.
12. **Database and Ticket Management** -- manages ticket records,
    predictions, historical information, and permitted agent feedback.

## Evaluation

The project report identifies the following evaluation metrics:

-   **Accuracy:** proportion of correctly classified samples.
-   **Precision:** proportion of predicted positive labels that are
    correct.
-   **Recall:** proportion of actual positive labels correctly
    identified.
-   **F1-score:** harmonic mean of precision and recall.

For multiclass classification, document whether macro, micro, or
weighted averaging is used. Evaluate each prediction target separately
and review class-wise performance, especially when the dataset is
imbalanced.

Add your final measured results and comparison charts here after
confirming them from your experiment outputs.

  Target                  Selected model     Accuracy   Precision   Recall   F1-score
  ----------------------- ---------------- ---------- ----------- -------- ----------
  Ticket Priority         To be filled            ---         ---      ---        ---
  Customer Satisfaction   To be filled            ---         ---      ---        ---
  Resolution              To be filled            ---         ---      ---        ---

## Technology

The project report explicitly identifies these core technologies and
methods:

-   **Natural Language Processing (NLP)**
-   **XLNet** for contextual feature extraction
-   **Machine Learning** classifiers: QDA, LDA, HGB, SGD, and Nearest
    Centroid
-   **NLTK / WordNet** for the preprocessing operations described in the
    report
-   **Label Encoding** for categorical target variables
-   **CSV** as the described input dataset format

The report does not specify a complete, verified application dependency
list or exact runtime versions. Add the actual libraries, versions,
database, and interface framework used by your implementation before
publishing installation instructions.

## Repository Structure

A suggested structure for organizing the source code is:

``` text
TicketSense-XL/
│
├── data/
│   └── README.md              # Dataset details and access instructions
│
├── notebooks/
│   └── analysis.ipynb         # Exploratory analysis (if included)
│
├── src/
│   ├── preprocessing.py       # Ticket text preprocessing
│   ├── feature_extraction.py  # XLNet embeddings
│   ├── train.py               # Classifier training
│   ├── predict.py             # Ticket predictions
│   └── evaluation.py          # Model evaluation
│
├── models/                    # Saved model artifacts (if included)
├── reports/                   # Charts and evaluation reports
├── app.py                     # Application entry point (if applicable)
├── requirements.txt           # Python dependencies
└── README.md
```

This is a suggested layout, not a claim about the exact files currently
present in the repository. Adjust it to match your actual project.

## Getting Started

The uploaded project report describes the system design and workflow but
does not provide verified repository-specific setup commands. Once the
source code is added, update this section with the actual environment
and entry point.

A typical Python setup may look like this:

``` bash
# Clone the repository
git clone <your-repository-url>

# Open the project directory
cd TicketSense-XL

# Create a virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS / Linux
source .venv/bin/activate

# Install dependencies (after creating requirements.txt)
pip install -r requirements.txt
```

Add the correct command to launch your application after confirming the
entry-point filename and framework. For example, the command may be
`python app.py` if your project uses that entry point.

### Dataset

The report describes loading ticket data from a CSV file. Place the
dataset in the location expected by your code and document:

-   Dataset name and source.
-   Required column names.
-   Target columns for priority, satisfaction, and resolution.
-   Any preprocessing or label-encoding requirements.
-   Applicable data-use and privacy restrictions.

Do not commit private customer tickets, credentials, or other sensitive
information to the repository.

## Using the Application

The report describes a GUI workflow with operations for:

1.  Dataset upload.
2.  Text preprocessing.
3.  Exploratory data analysis (EDA).
4.  XLNet feature extraction.
5.  Classifier training (QDA, LDA, SGD, NC, and HGB).
6.  Ticket prediction.
7.  Model comparison tables.
8.  Review of predicted priority, customer satisfaction, and resolution.

The exact controls and steps may vary according to the final application
implementation.

## Future Enhancements

-   Improve dynamic clustering of related support incidents.
-   Extend organization-specific model customization.
-   Integrate and maintain a structured knowledge base.
-   Improve proactive identification of recurring or escalating
    incidents.
-   Add confidence-based review and feedback workflows.
-   Expand dashboard and reporting capabilities.
-   Evaluate the system on additional datasets and real-world ticket
    distributions.

## Project Team

**Department of Computer Science and Engineering**\
**Kommuri Pratap Reddy Institute of Technology (Autonomous)**

  Team member           Roll number
  --------------------- -------------
  B. Sai Prasanna       23RA1A05C6
  G. Sai Krishna        23RA1A05C2
  M. Sai Vivek          23RA1A05C8
  B. Sai Pranay Kumar   23RA1A05C5

**Project Guide:** Dr. K. Vamshee Krishna\
**Academic Program:** B.Tech -- Computer Science and Engineering\
**Batch:** 2023--2027

## Acknowledgements

We express our sincere gratitude to our project guide, Dr. K. Vamshee
Krishna, the Department of Computer Science and Engineering, and Kommuri
Pratap Reddy Institute of Technology for their guidance and support
throughout the project.

------------------------------------------------------------------------

**If you use or extend this project, please cite the project
appropriately and follow the license and data-use terms applicable to
the code and datasets.**