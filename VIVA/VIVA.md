# Fake News Detector Viva Questions and Answers

This file is written step by step for viva preparation. Pehle training script aur data/model flow, phir Django backend, API, prediction, aur deployment-related questions cover kiye gaye hain.

## 1) `train_model.py` Data aur Model Section

### 1. Project ka main purpose kya hai?
**Answer:** Is project ka purpose news article ko fake ya real classify karna hai. User text ya URL deta hai, system us article ka analysis karke prediction return karta hai.

### 2. `train_model.py` ka role kya hai?
**Answer:** `train_model.py` model training ke liye use hota hai. Ye dataset load karta hai, preprocess karta hai, tokenization karta hai, DistilBERT train karta hai, metrics evaluate karta hai, aur trained model save karta hai.

### 3. Is training script me kaunsa model use hua hai?
**Answer:** DistilBERT use hua hai, specifically `DistilBertForSequenceClassification`.

### 4. DistilBERT kyun use kiya gaya?
**Answer:** DistilBERT BERT ka lighter aur faster version hai. Ye less memory leta hai aur CPU par bhi relatively better run karta hai, isliye small or medium projects ke liye suitable hai.

### 5. Training data kaha se aata hai?
**Answer:** Data `dataset/Fake.csv` aur `dataset/True.csv` files se aata hai.

### 6. Fake aur real news ko kaise label kiya gaya hai?
**Answer:** Fake news ko label `0` diya gaya hai aur real news ko label `1`.

### 7. Dataset ko merge kaise kiya gaya hai?
**Answer:** Fake aur real CSV ko pandas ke through concatenate karke ek combined dataframe banaya gaya hai.

### 8. Dataset preprocessing me kya steps hote hain?
**Answer:** Empty rows remove ki jaati hain, text column clean ki jaati hai, strip kiya jaata hai, aur blank text rows hata di jaati hain.

### 9. `text` column ka importance kya hai?
**Answer:** `text` column article ka main content hota hai. Model isi text se learning karta hai aur classification karta hai.

### 10. `labels` column ka use kya hai?
**Answer:** `labels` target variable hai. Ye batata hai ki article fake hai ya real.

### 11. Train-test split ka kya use hai?
**Answer:** Data ko training aur validation set me divide kiya jaata hai taaki model ko train bhi kiya ja sake aur unseen data par evaluate bhi kiya ja sake.

### 12. Validation split kitna hai?
**Answer:** `test_size=0.2` use hua hai, yani 20% data validation ke liye aur 80% training ke liye.

### 13. `stratify=data["labels"]` kyun use kiya gaya hai?
**Answer:** Isse train aur validation split me class distribution balanced rehti hai. Fake aur real classes ka proportion preserve hota hai.

### 14. Tokenization kya hoti hai?
**Answer:** Tokenization text ko numeric tokens me convert karne ki process hai, jise model samajh sake.

### 15. `tokenize_batch()` kya karta hai?
**Answer:** Ye text batch ko tokenizer se tokenize karta hai, truncation ke saath aur `max_length=256` ke limit me.

### 16. `max_length=256` kyun rakha gaya hai?
**Answer:** Long articles ko limit karne ke liye. Isse training faster hoti hai aur memory usage control me rehta hai.

### 17. `truncation=True` ka kya matlab hai?
**Answer:** Agar input text zyada lamba hai, to extra part cut ho jaata hai.

### 18. `Dataset.from_pandas()` ka use kya hai?
**Answer:** Pandas dataframe ko Hugging Face `Dataset` format me convert karne ke liye use hota hai.

### 19. `set_format("torch")` kyun use hota hai?
**Answer:** Taaki dataset direct PyTorch tensors me convert ho aur training pipeline me use ho sake.

### 20. Model me `num_labels=2` kyun hai?
**Answer:** Kyunki problem binary classification hai: fake ya real.

### 21. `id2label` aur `label2id` ka use kya hai?
**Answer:** Ye class IDs aur human-readable labels ke beech mapping provide karte hain.

### 22. Class weights kyun calculate ki gayi hain?
**Answer:** Agar dataset imbalance ho to model ek class ki taraf zyada biased na ho. Class weights minority class ko better importance deti hain.

### 23. `WeightedTrainer` kya hai?
**Answer:** Ye custom trainer class hai jo weighted loss use karta hai, especially imbalanced data ke liye.

### 24. `compute_loss()` override kyun kiya gaya?
**Answer:** Default loss ke bajay class-weighted CrossEntropyLoss use karne ke liye.

### 25. CrossEntropyLoss kya hota hai?
**Answer:** Ye classification tasks me use hone wala loss function hai jo predicted logits aur true labels ke beech error measure karta hai.

### 26. `compute_metrics()` kya return karta hai?
**Answer:** Accuracy, precision, recall, aur F1-score return karta hai.

### 27. Accuracy kya batati hai?
**Answer:** Total predictions me se kitne sahi the.

### 28. Precision kya batati hai?
**Answer:** Jitne fake/real predictions model ne diye, unme se kitne correct the.

### 29. Recall kya batati hai?
**Answer:** Actual class ke kitne examples model ne correctly identify kiye.

### 30. F1-score kyun important hai?
**Answer:** Jab precision aur recall dono ko balance karna ho, tab F1-score useful hota hai.

### 31. `TrainingArguments` ka use kya hai?
**Answer:** Model training ke hyperparameters aur training behavior define karne ke liye.

### 32. `num_train_epochs=3` kyun rakha gaya hai?
**Answer:** Model ko 3 passes me train kiya gaya hai taaki overfitting aur training time ke beech balance rahe.

### 33. `learning_rate=2e-5` ka meaning kya hai?
**Answer:** Ye optimizer ka step size hai. Small learning rate transformer models ke liye safe aur effective hota hai.

### 34. `warmup_steps=500` ka use kya hai?
**Answer:** Training ke start me learning rate gradually increase hoti hai, jisse training stable rehti hai.

### 35. `weight_decay=0.01` kya karta hai?
**Answer:** Overfitting reduce karne me help karta hai by penalizing large weights.

### 36. `gradient_accumulation_steps=2` kyun use hua hai?
**Answer:** Effective batch size badhane ke liye, especially jab memory limited ho.

### 37. `EarlyStoppingCallback` kyun use kiya gaya hai?
**Answer:** Agar validation performance improve na ho to training jaldi stop ho jaaye, taaki overfitting na ho.

### 38. `metric_for_best_model="f1"` kyun rakha gaya hai?
**Answer:** Best model ko F1-score ke basis par select kiya jaata hai, kyunki binary classification me ye balanced metric hai.

### 39. `load_best_model_at_end=True` ka benefit kya hai?
**Answer:** Training ke end par automatically best validation model load ho jaata hai.

### 40. `save_strategy="epoch"` kya karta hai?
**Answer:** Har epoch ke baad model save karta hai.

### 41. `save_total_limit=2` kyun hai?
**Answer:** Purane checkpoints ka number limit karne ke liye, storage bachane ke liye.

### 42. `fp16=False` kyun hai?
**Answer:** CPU stability ke liye float32 use ki gayi hai.

### 43. `dataloader_pin_memory=False` kyun set hai?
**Answer:** CPU environment me pin memory ka useful hona kam hota hai, isliye stability ke liye off hai.

### 44. `remove_unused_columns=False` ka use kya hai?
**Answer:** Training ke dauran required columns ko accidentally drop hone se bachata hai.

### 45. `optim="adamw_torch"` kya hota hai?
**Answer:** Ye stable AdamW optimizer implementation hai jo transformer training me commonly use hota hai.

### 46. `set_seed(42)` ka use kya hai?
**Answer:** Reproducible results ke liye seed fix ki jaati hai.

### 47. Model training ke baad kya save hota hai?
**Answer:** Trained model aur tokenizer dono `bert_model/` folder me save hote hain.

### 48. `trainer.save_model()` kya karta hai?
**Answer:** Fine-tuned model files ko disk par store karta hai.

### 49. `tokenizer.save_pretrained()` kyun use hota hai?
**Answer:** Same tokenizer later inference ke time reuse karne ke liye save hota hai.

### 50. `results/` aur `logs/` folders ka use kya hai?
**Answer:** `results/` training outputs aur checkpoints ke liye, aur `logs/` training logs ke liye use hota hai.

## 2) Training Pipeline Viva Questions

### 51. Model training pipeline ka flow kya hai?
**Answer:** Dataset load → clean → split → tokenize → model load → weighted training → evaluation → save model.

### 52. Agar dataset me imbalance ho to kya problem hoti hai?
**Answer:** Model majority class ki taraf biased ho sakta hai aur minority class ko ignore kar sakta hai.

### 53. Imbalanced dataset ka solution kya hai?
**Answer:** Class weights, oversampling, undersampling, ya focal loss use ki ja sakti hai.

### 54. DistilBERT ka output kya hota hai?
**Answer:** Ye logits deta hai, jisse fake aur real classes ke probabilities derive ki jaati hain.

### 55. `np.argmax(logits, axis=-1)` kya karta hai?
**Answer:** Highest score wali class choose karta hai.

### 56. Why use Hugging Face `Trainer`?
**Answer:** Ye training, evaluation, logging, checkpointing, aur metric handling ko easy bana deta hai.

### 57. Why custom trainer instead of default trainer?
**Answer:** Weighted loss use karne ke liye custom trainer banaya gaya hai.

### 58. `train_model.py` me CPU support ka kya dhyan rakha gaya hai?
**Answer:** `fp16=False`, `pin_memory=False`, aur DistilBERT ka lightweight nature CPU compatibility improve karte hain.

## 3) Django Backend aur Database Questions

### 59. Is project me backend ka framework kya hai?
**Answer:** Django use hua hai, aur API layer ke liye Django REST Framework use hua hai.

### 60. `Prediction` model kya store karta hai?
**Answer:** User, text, result, confidence, verification_result, aur created_at.

### 61. `verification_result` field kis type ka hai?
**Answer:** Ye JSONField hai.

### 62. JSONField kyun use kiya gaya hai?
**Answer:** Verification se related detailed structured data store karne ke liye.

### 63. `user` field nullable kyun hai?
**Answer:** Anonymous users bhi predictions kar sakte hain, isliye user optional rakha gaya hai.

### 64. `__str__()` method ka role kya hai?
**Answer:** Ye model object ka readable string representation deta hai.

### 65. Serializer ka kya role hai?
**Answer:** Model data ko JSON me convert karne aur API responses me use karne ke liye.

### 66. `PredictionSerializer` me `fields='__all__'` kyun use hua hai?
**Answer:** Taaki Prediction model ke saare fields API me expose ho saken.

## 4) API aur Prediction Flow Questions

### 67. Main prediction endpoint kaunsa hai?
**Answer:** `POST /api/predict/`

### 68. Prediction API me input kya ho sakta hai?
**Answer:** Direct text ya URL.

### 69. URL input se text kaise nikala jaata hai?
**Answer:** `extract_text_from_url()` utility use hoti hai.

### 70. `_looks_like_url()` function ka kya purpose hai?
**Answer:** Ye check karta hai ki input URL jaisa lag raha hai ya nahi.

### 71. Agar URL extraction fail ho jaye to kya hota hai?
**Answer:** Helpful error message return hota hai, aur agar manual text available ho to processing continue ho sakti hai.

### 72. Prediction ke baad history kaise store hoti hai?
**Answer:** Logged-in user ke liye `Prediction` object database me create hota hai.

### 73. `prediction_history` endpoint kya karta hai?
**Answer:** Logged-in user ki purani predictions return karta hai.

### 74. Authentication endpoints ka role kya hai?
**Answer:** User registration, login, logout, aur current user status handle karte hain.

### 75. `auth_me` endpoint kya return karta hai?
**Answer:** User authenticated hai ya nahi, aur authenticated ho to user details.

## 5) Verification aur Hybrid Logic Questions

### 76. Is project me sirf ML model use hua hai ya aur bhi verification hai?
**Answer:** ML model ke saath verification layer bhi hai, jisse final result aur robust banta hai.

### 77. `verification_result` me kya kya store hota hai?
**Answer:** Decision reason, signal score, primary prediction, provider info, confidence, explanation, errors, aur detailed verification results.

### 78. `signal_score` kya hota hai?
**Answer:** Ye sensational language ya suspicious patterns based heuristic score hota hai.

### 79. Signal score ka use kyun hota hai?
**Answer:** Obvious fake-news style text ko extra caution ke saath handle karne ke liye.

### 80. Confidence ka role kya hai?
**Answer:** Confidence batata hai ki model prediction par kitna sure hai.

## 6) Practical Viva Questions

### 81. Agar model load na ho to kya hoga?
**Answer:** Prediction flow me fallback handling hoti hai, aur error response aa sakta hai.

### 82. Agar API key available na ho to kya hoga?
**Answer:** Verification status disabled ya partial ho sakta hai, lekin primary model inference phir bhi kaam kar sakta hai.

### 83. Is project ka main advantage kya hai?
**Answer:** Ye sirf classifier nahi hai, balki text/URL support, training pipeline, confidence output, aur history tracking bhi deta hai.

### 84. Is project ki limitation kya ho sakti hai?
**Answer:** Training data quality, language variation, breaking news context, aur URL extraction failures model performance ko affect kar sakte hain.

### 85. Future improvement kya ho sakte hain?
**Answer:** Multi-language support, larger dataset, better ensemble strategies, better explainability, aur deployment optimization.

## 7) Short Revision Answers

### 86. Fake news detection kya hai?
**Answer:** News article ko fake ya real classify karna.

### 87. Binary classification kya hoti hai?
**Answer:** Jab sirf do classes ho.

### 88. DistilBERT kya hai?
**Answer:** BERT ka compact version.

### 89. F1-score kya hai?
**Answer:** Precision aur recall ka harmonic mean.

### 90. JSONField kya hai?
**Answer:** Structured JSON data store karne ka field.

### 91. Early stopping kyun use hota hai?
**Answer:** Overfitting kam karne ke liye.

### 92. Tokenization kya deti hai?
**Answer:** Text ka numeric representation.

### 93. Classification task me logits kya hote hain?
**Answer:** Raw model outputs before softmax.

### 94. Why save tokenizer with model?
**Answer:** Inference ke time same preprocessing maintain karne ke liye.

### 95. Prediction endpoint ka input kya hai?
**Answer:** Text ya URL.

### 96. Prediction history kis ke liye hai?
**Answer:** Logged-in users ke liye.

### 97. Anonymous user predictions kaise handle hoti hain?
**Answer:** Prediction database me user null reh sakta hai.

### 98. Why use SQLite in this project?
**Answer:** Simple local development aur testing ke liye.

### 99. Model output kis form me aata hai?
**Answer:** Fake/real probability-based classification output.

### 100. Is project ko viva me kaise present karein?
**Answer:** Pehle problem statement, phir dataset, model training, evaluation, backend APIs, prediction flow, aur future scope explain karein.

## 8) Viva Presentation Order

1. Problem statement
2. Dataset source aur labeling
3. `train_model.py` me preprocessing
4. DistilBERT architecture aur training
5. Metrics aur validation
6. Django model aur database
7. Prediction API flow
8. URL handling aur extraction
9. Verification and confidence logic
10. Future scope

## 9) One-Line Summary

Ye project fake news detection ke liye DistilBERT based training pipeline, Django backend, prediction API, authentication, aur history tracking combine karta hai.