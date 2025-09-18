import json
from django.shortcuts import render
from django.http import JsonResponse
from .utils.test import get_solar_forecast
from django.views.decorators.csrf import csrf_exempt

def home(request):
    return render(request, 'hackovate-dashboard2.html')

@csrf_exempt
def submitForm(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))

            surface_area = float(data.get("surfaceArea", 10))
            tilt_angle = float(data.get("tiltAngle", 15))
            azimuth_angle = float(data.get("azimuthAngle", 180))
            lat = float(data.get("lat"))
            lon = float(data.get("lon"))

            api_key = "3134de8bea3eaf3080a5306d29513ddd"

            forecast = get_solar_forecast(lat, lon, surface_area, tilt_angle, azimuth_angle, api_key)
            # ✅ Return JSON instead of rendering template
            return JsonResponse({
                "status": "success",
                "data": forecast
            }, status=200)

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

# def solar(request):
#     return render(request, 'solar.html')
