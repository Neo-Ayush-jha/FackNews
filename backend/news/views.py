from django.contrib.auth import authenticate, get_user_model, login, logout
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Prediction
from .serializers import PredictionSerializer
from .utils import analyze_text, extract_text_from_url


@ensure_csrf_cookie
def home_page(request):
    return render(request, "news/home.html")


def results_page(request):
    return render(request, "news/results.html")


@ensure_csrf_cookie
def about_page(request):
    return render(request, "news/about.html")


@ensure_csrf_cookie
def auth_page(request):
    return render(request, "news/auth.html")


@ensure_csrf_cookie
def upload_page(request):
    return render(request, "news/upload.html")


@api_view(["POST"])
def register_user(request):
    name = request.data.get("name", "").strip()
    email = request.data.get("email", "").strip().lower()
    password = request.data.get("password", "")

    if not email or not password:
        return Response({"error": "Email and password are required."}, status=400)
    if len(password) < 6:
        return Response({"error": "Password must be at least 6 characters."}, status=400)

    user_model = get_user_model()
    if user_model.objects.filter(username=email).exists():
        return Response({"error": "User already exists."}, status=400)

    user = user_model.objects.create_user(username=email, email=email, password=password)
    if name:
        user.first_name = name
        user.save(update_fields=["first_name"])

    return Response({
        "message": "Registration successful",
        "user": {"email": user.email, "name": user.first_name or user.username},
    })


@api_view(["POST"])
def login_user(request):
    email = request.data.get("email", "").strip().lower()
    password = request.data.get("password", "")

    if not email or not password:
        return Response({"error": "Email and password are required."}, status=400)

    user = authenticate(request, username=email, password=password)
    if not user:
        return Response({"error": "Invalid credentials."}, status=400)

    login(request, user)
    return Response({
        "message": "Login successful",
        "user": {"email": user.email, "name": user.first_name or user.username},
    })


@api_view(["POST"])
def logout_user(request):
    logout(request)
    return Response({"message": "Logged out"})


@api_view(["GET"])
def auth_me(request):
    if not request.user.is_authenticated:
        return Response({"authenticated": False})
    return Response({
        "authenticated": True,
        "user": {"email": request.user.email, "name": request.user.first_name or request.user.username},
    })

@api_view(["POST"])
def predict_news(request):
    if not request.user.is_authenticated:
        return Response({"error": "Authentication required."}, status=401)

    text = request.data.get("text", "")
    url = request.data.get("url", "")

    if url:
        try:
            text = extract_text_from_url(url)
        except Exception as exc:
            return Response({"error": f"Could not extract article text: {str(exc)}"}, status=400)

    if not text:
        return Response({"error": "No text provided"}, status=400)

    try:
        analysis = analyze_text(text)
    except Exception as exc:
        return Response({"error": f"Prediction failed: {str(exc)}"}, status=500)

    Prediction.objects.create(
        user=request.user,
        text=text,
        result=analysis["prediction"],
        confidence=analysis["confidence"],
        verification_result={
            "decision_reason": analysis["decision_reason"],
            "signal_score": analysis["signal_score"],
            "primary_prediction": analysis["primary_prediction"],
            "primary_confidence": analysis["primary_confidence"],
            "provider": analysis["verification_provider"],
            "model": analysis["verification_model"],
            "status": analysis["verification_status"],
            "prediction": analysis["verification_prediction"],
            "confidence": analysis["verification_confidence"],
            "explanation": analysis["verification_explanation"],
            "error": analysis["verification_error"],
        },
    )

    return Response({
        "prediction": analysis["prediction"],
        "confidence": analysis["confidence"],
        "decision_reason": analysis["decision_reason"],
        "signal_score": analysis["signal_score"],
        "primary_prediction": analysis["primary_prediction"],
        "primary_confidence": analysis["primary_confidence"],
        "verification_status": analysis["verification_status"],
        "verification_provider": analysis["verification_provider"],
        "verification_model": analysis["verification_model"],
        "verification_prediction": analysis["verification_prediction"],
        "verification_confidence": analysis["verification_confidence"],
        "verification_explanation": analysis["verification_explanation"],
        "verification_error": analysis["verification_error"],
    })


@api_view(["GET"])
def prediction_history(request):
    if not request.user.is_authenticated:
        return Response({"error": "Authentication required."}, status=401)

    predictions = Prediction.objects.filter(user=request.user).order_by("-created_at")
    serializer = PredictionSerializer(predictions, many=True)
    return Response(serializer.data)
