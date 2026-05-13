# train_model.py Viva Questions

## 1. train_model.py ka main role kya hai?
Answer: Ye script model training ke liye use hoti hai. Isme dataset load hota hai, preprocess hota hai, tokenization hoti hai, DistilBERT train hota hai, aur final model save hota hai.

## 2. Is script me kaunsa model use hua hai?
Answer: DistilBERT sequence classification model use hua hai.

## 3. Data kaha se aata hai?
Answer: Dataset folder me maujood Fake.csv aur True.csv se.

## 4. Fake aur real labels ka mapping kya hai?
Answer: Fake news ko 0 aur real news ko 1 label diya gaya hai.

## 5. Dataset preprocessing me kya steps hain?
Answer: Missing rows remove hoti hain, text clean hota hai, blank text hata diya jata hai, aur labels ke saath combined dataframe banaya jata hai.

## 6. Tokenization kyun zaroori hai?
Answer: Model ko raw text samajh nahi aata, isliye text ko numeric tokens me convert karna padta hai.

## 7. max_length 256 kyun rakha gaya hai?
Answer: Long articles ko limit karne aur training ko fast aur stable rakhne ke liye.

## 8. Class weights kyun use hui hain?
Answer: Imbalanced dataset me minority class ko better importance dene ke liye.

## 9. Custom trainer kyun banaya gaya hai?
Answer: Weighted CrossEntropyLoss use karne ke liye.

## 10. Kaun si metrics calculate hoti hain?
Answer: Accuracy, precision, recall, aur F1-score.

## 11. Early stopping ka kya benefit hai?
Answer: Overfitting kam hota hai aur validation performance kharab hone par training ruk sakti hai.

## 12. Model aur tokenizer kaha save hote hain?
Answer: bert_model folder me.
