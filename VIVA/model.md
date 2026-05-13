# model.py Viva Questions

## 1. Prediction model ka use kya hai?
Answer: Prediction ka record database me save karna.

## 2. Is model me kaun se fields hain?
Answer: user, text, result, confidence, verification_result, created_at.

## 3. user nullable kyun hai?
Answer: Anonymous predictions ke liye.

## 4. result field kya store karta hai?
Answer: Fake News ya Real News.

## 5. confidence field ka role kya hai?
Answer: Prediction ki surety dikhata hai.

## 6. verification_result kya hai?
Answer: Extra analysis ka JSON data.

## 7. created_at kyun important hai?
Answer: Prediction time store karta hai.

## 8. JSONField kyun use hua hai?
Answer: Structured data save karne ke liye.

## 9. related_name ka use kya hai?
Answer: User se predictions access karne ke liye.

## 10. __str__ method kya deta hai?
Answer: Readable model name.
