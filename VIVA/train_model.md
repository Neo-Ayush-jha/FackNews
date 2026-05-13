# train_model.py Viva Questions

## 1. train_model.py ka kaam kya hai?
Answer: Model train karna, evaluate karna, aur save karna.

## 2. Kaunsa model use hua hai?
Answer: DistilBERT sequence classification.

## 3. Training data kahan se aata hai?
Answer: Fake.csv aur True.csv se.

## 4. Labels ka mapping kya hai?
Answer: Fake = 0, Real = 1.

## 5. Tokenization kyun hoti hai?
Answer: Text ko model-readable tokens me convert karne ke liye.

## 6. max_length 256 kyun hai?
Answer: Long text ko limit karne aur training fast rakhne ke liye.

## 7. Class weights kyun use hui hain?
Answer: Imbalanced data ko handle karne ke liye.

## 8. Custom trainer kya karta hai?
Answer: Weighted loss use karta hai.

## 9. Kaun si metrics nikalti hain?
Answer: Accuracy, precision, recall, aur F1.

## 10. Early stopping ka fayda kya hai?
Answer: Overfitting kam hota hai.

## 11. Model kahan save hota hai?
Answer: bert_model folder me.

## 12. Seed 42 kyun use hua hai?
Answer: Reproducible results ke liye.
