from django.contrib.auth import authenticate, get_user_model, login, logout
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Prediction
from .serializers import PredictionSerializer


def _looks_like_url(value):
    value = (value or "").strip().lower()
    return (
        value.startswith(("http://", "https://", "www."))
        or ((" " not in value) and "." in value)
    )


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
    text = (request.data.get("text") or "").strip()
    url = (request.data.get("url") or "").strip()

    if url:
        if _looks_like_url(url):
            try:
                from .utils import extract_text_from_url

                text = extract_text_from_url(url)
            except Exception as exc:
                error_msg = str(exc)
                # Provide helpful suggestions
                if "blocked" in error_msg.lower() or "fetch" in error_msg.lower():
                    error_msg += " You can try: 1) Copy and paste the article text directly, 2) Try a mobile version of the URL, 3) Check if the URL is correct"
                # Allow manual pasted text to continue if URL extraction is blocked.
                if not text:
                    return Response({"error": f"Could not extract article text: {error_msg}"}, status=400)
        else:
            text = f"{text}\n\n{url}".strip()

    if not text:
        return Response({"error": "No text provided"}, status=400)

    try:
        from .utils import analyze_text

        analysis = analyze_text(text)
    except Exception as exc:
        return Response({"error": f"Prediction failed: {str(exc)}"}, status=500)

    if request.user.is_authenticated:
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
                "results": analysis.get("verification_results", []),
                "gemini_result": analysis.get("gemini_result"),
                "groq_result": analysis.get("groq_result"),
                "primary_model_status": analysis.get("primary_model_status"),
                "primary_model_note": analysis.get("primary_model_note"),
            },
        )

    return Response({
        "prediction": analysis["prediction"],
        "confidence": analysis["confidence"],
        "text": text,
        "decision_reason": analysis["decision_reason"],
        "signal_score": analysis["signal_score"],
        "primary_prediction": analysis["primary_prediction"],
        "primary_confidence": analysis["primary_confidence"],
        "primary_fake_confidence": analysis.get("primary_fake_confidence"),
        "primary_real_confidence": analysis.get("primary_real_confidence"),
        "primary_model_status": analysis.get("primary_model_status"),
        "primary_model_note": analysis.get("primary_model_note"),
        "verification_status": analysis["verification_status"],
        "verification_provider": analysis["verification_provider"],
        "verification_model": analysis["verification_model"],
        "verification_prediction": analysis["verification_prediction"],
        "verification_confidence": analysis["verification_confidence"],
        "verification_explanation": analysis["verification_explanation"],
        "verification_error": analysis["verification_error"],
        "verification_results": analysis.get("verification_results", []),
        "gemini_result": analysis.get("gemini_result"),
        "groq_result": analysis.get("groq_result"),
    })


@api_view(["GET"])
def prediction_history(request):
    if not request.user.is_authenticated:
        return Response([])

    predictions = Prediction.objects.filter(user=request.user).order_by("-created_at")
    serializer = PredictionSerializer(predictions, many=True)
    return Response(serializer.data)


@api_view(["GET"])
def search_news_api(request):
    """
    API endpoint to search news from multiple sources.
    
    Query parameters:
        query: Search query (required)
        source: News source - 'all', 'timesofindia', 'ndtv', 'hindustantimes', 'bbc', 'reuters', etc.
        limit: Number of results (default: 5, max: 20)
    """
    query = request.GET.get("query", "").strip()
    source = request.GET.get("source", "all").strip().lower()
    limit = min(int(request.GET.get("limit", 5)), 20)
    
    if not query:
        return Response({"error": "Search query is required"}, status=400)
    
    if len(query) > 200:
        return Response({"error": "Search query is too long"}, status=400)
    
    try:
        from .utils import search_news
        results = search_news(query, source=source, limit=limit)
        return Response({
            "query": query,
            "source": source,
            "count": len(results),
            "results": results
        })
    except Exception as exc:
        return Response({"error": f"News search failed: {str(exc)}"}, status=500)
