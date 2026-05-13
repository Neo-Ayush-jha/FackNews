# model.py / models.py Viva Questions

## 1. Prediction model ka use kya hai?
Answer: Har prediction ka record database me save karne ke liye.

## 2. Prediction model me kaun se fields hain?
Answer: user, text, result, confidence, verification_result, aur created_at.

## 3. user field nullable kyun hai?
Answer: Anonymous users ke liye bhi prediction allow karne ke liye.

## 4. text field kya store karta hai?
Answer: Article ka full text ya analyzed input.

## 5. result field kya store karta hai?
Answer: Final label jaise Fake News ya Real News.

## 6. confidence field kya batata hai?
Answer: Prediction ki surety percentage.

## 7. verification_result field ka use kya hai?
Answer: Detailed JSON analysis save karne ke liye.

## 8. created_at ka kya role hai?
Answer: Prediction kab create hui thi, ye timestamp store karta hai.

## 9. JSONField kyun use hua hai?
Answer: Structured nested data save karne ke liye.

## 10. __str__ method ka fayda kya hai?
Answer: Admin aur shell me readable representation milti hai.
