# Fake News Detector Project Report

## 1. Project Overview

**Project Title:** Fake News Detector  
**Project Type:** AI-based Web Application  
**Purpose:** To detect whether a news article is real or fake by using a trained NLP model and presenting results through a user-friendly web interface.

This project is currently built with a Django backend and a machine learning inference layer based on DistilBERT. For future enhancement, the frontend is planned in **React** and the database is planned in **PostgreSQL** to make the system more scalable, modern, and production-ready.

## 2. Proposed Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React |
| Backend | Django + Django REST Framework |
| AI/ML Model | DistilBERT |
| Database | PostgreSQL |
| API Style | REST API |
| Authentication | Django Authentication |

## 3. Main Modules

1. **User Interface**
   Accepts news text or article URL from the user.

2. **Authentication Module**
   Handles user registration, login, logout, and session checking.

3. **Prediction Module**
   Sends input text to the backend and returns fake/real prediction with confidence.

4. **History Module**
   Stores and displays previous prediction records.

5. **Database Module**
   Stores users and prediction data in PostgreSQL.

6. **ML Inference Module**
   Loads the trained DistilBERT model and performs news classification.

## 4. Current and Planned Architecture

- **Current system:** Django templates + Django APIs + local ML model + SQLite
- **Planned system:** React frontend + Django REST API backend + DistilBERT model + PostgreSQL

The planned architecture improves separation of concerns, frontend flexibility, and database scalability.

## 5. Data Flow Diagram (DFD)

### Level 0 DFD

```mermaid
flowchart LR
    U[User] --> S[Fake News Detection System]
    S --> U
    S --> DB[(PostgreSQL Database)]
    S --> ML[ML Model]
```

### Level 1 DFD

```mermaid
flowchart TD
    U[User] --> P1[Submit News Text / URL]
    P1 --> P2[React Frontend]
    P2 --> P3[Django REST API]
    P3 --> P4[Text Extraction and Validation]
    P4 --> P5[ML Prediction Engine]
    P5 --> P6[Store Prediction]
    P6 --> DB[(PostgreSQL)]
    P5 --> P2
    P2 --> U
```

## 6. UML Use Case Diagram

```mermaid
flowchart LR
    A[User]
    B[Admin]

    UC1((Register))
    UC2((Login))
    UC3((Submit News Text))
    UC4((Submit News URL))
    UC5((View Prediction))
    UC6((View History))
    UC7((Manage System))

    A --> UC1
    A --> UC2
    A --> UC3
    A --> UC4
    A --> UC5
    A --> UC6
    B --> UC7
```

## 7. Sequence Diagram (Scan Request Flow)

```mermaid
sequenceDiagram
    actor User
    participant React as React Frontend
    participant API as Django API
    participant Extract as Text Extractor
    participant Model as DistilBERT Model
    participant DB as PostgreSQL

    User->>React: Enter text or URL and click Scan
    React->>API: POST /api/predict
    API->>Extract: Extract text if URL provided
    Extract-->>API: Clean article text
    API->>Model: Predict news class
    Model-->>API: Fake/Real + confidence
    API->>DB: Save prediction record
    DB-->>API: Saved successfully
    API-->>React: Prediction response
    React-->>User: Show result and confidence
```

## 8. Activity Diagram

```mermaid
flowchart TD
    A([Start]) --> B[Open Application]
    B --> C[Login or Continue]
    C --> D[Enter News Text or URL]
    D --> E[Validate Input]
    E -->|Valid| F[Send Request to Backend]
    E -->|Invalid| G[Show Error]
    G --> D
    F --> H[Run ML Prediction]
    H --> I[Store Prediction in Database]
    I --> J[Return Result]
    J --> K[Display Prediction]
    K --> L([End])
```

## 9. ER Diagram

```mermaid
erDiagram
    USER ||--o{ PREDICTION : creates

    USER {
        int id
        string name
        string email
        string password
    }

    PREDICTION {
        int id
        text news_text
        string result
        float confidence
        datetime created_at
        int user_id
    }
```

## 10. Flow Control Diagram

```mermaid
flowchart LR
    A[Input Received] --> B{Is URL Provided?}
    B -->|Yes| C[Extract Article Text]
    B -->|No| D[Use Direct Text]
    C --> E[Clean and Validate Text]
    D --> E
    E --> F[Run DistilBERT Prediction]
    F --> G{Prediction Result}
    G -->|Fake| H[Show Fake News]
    G -->|Real| I[Show Real News]
    H --> J[Save to Database]
    I --> J
```

## 11. System Diagram

```mermaid
flowchart TB
    U[User]
    FE[React Frontend]
    API[Django REST API]
    AUTH[Authentication Module]
    PRED[Prediction Module]
    HIST[History Module]
    MODEL[DistilBERT Inference Engine]
    DB[(PostgreSQL Database)]

    U --> FE
    FE --> API
    API --> AUTH
    API --> PRED
    API --> HIST
    PRED --> MODEL
    AUTH --> DB
    HIST --> DB
    PRED --> DB
```

## 12. Conclusion

The Fake News Detector project is a useful AI-based application for identifying misleading news content. The backend already supports prediction, authentication, and history features. By integrating **React** for the frontend and **PostgreSQL** for the database, the project can become more interactive, scalable, and better suited for real-world deployment. The proposed diagrams represent the overall structure, data movement, and system behavior in a simple and effective way.
