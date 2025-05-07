from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from io import BytesIO
import base64
import traceback
import requests

from ..styletransfer.tasks import wait_for_result


# def image_upload_view(request):
#     if request.method == "POST":
#         style_file = request.FILES.get("style")
#         content_file = request.FILES.get("content")
#
#         if not style_file or not content_file:
#             return render(request, "upload.html", {"error": "Style and content files are required."})
#
#         # style_path = default_storage.save(f"uploads/{uuid.uuid4()}_{style_file.name}", style_file)
#         # content_path = default_storage.save(f"uploads/{uuid.uuid4()}_{content_file.name}", content_file)
#         #
#         # full_style_path = os.path.join(settings.MEDIA_ROOT, style_path)
#         # full_content_path = os.path.join(settings.MEDIA_ROOT, content_path)
#
#         result_buf = wait_for_result(
#             content=content_file,
#             style=style_file,
#             prompt="",
#             preprocessor="Contour"  # or "Lineart"
#         )
#
#         if not result_buf:
#             return render(request, "upload.html")
#
#         # base64로 인코딩
#         img_base64 = base64.b64encode(result_buf.getvalue()).decode('utf-8')
#         img_data_url = f"data:image/png;base64,{img_base64}"
#         result_buf.close()
#         if not img_base64:
#             traceback.print_exc()
#             return render(request, "upload.html")
#         return render(request, "upload.html", {"result_data_url": img_data_url})
#     print()
#     return render(request, "upload.html")

# @api_view(['POST'])
# def image_upload_view(request):
#     style_file = request.FILES.get("style")
#     content_file = request.FILES.get("content")
#
#     if not style_file or not content_file:
#         return Response({"error": "Style and content files are required."}, status=status.HTTP_400_BAD_REQUEST)
#
#     result_buf = wait_for_result(
#         content=content_file,
#         style=style_file,
#         prompt="",
#         preprocessor="Contour"
#     )
#
#     if not result_buf:
#         return Response({"error": "Image generation failed or timed out."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#
#     img_base64 = base64.b64encode(result_buf.getvalue()).decode('utf-8')
#     result_buf.close()
#
#     return Response({
#         "image_base64": img_base64,
#         "mime_type": "image/png"
#     })
# @api_view(['POST'])
# def image_upload_view(request):
#     try:
#         style_b64 = request.data.get("style")
#         content_b64 = request.data.get("content")
#         if not style_b64 or not content_b64:
#             return Response({"error": "Base64-encoded style and content images are required."}, status=status.HTTP_400_BAD_REQUEST)
#
#         # base64 디코딩
#         style_data = base64.b64decode(style_b64.split(",")[-1])  # 'data:image/png;base64,...' 형식이면 콤마 뒤만 사용
#         content_data = base64.b64decode(content_b64.split(",")[-1])
#
#         style_file = BytesIO(style_data)
#         content_file = BytesIO(content_data)
#
#         # test_image_path = "./media/outputs/65b5f9f6-87c2-4473-b1e3-0a6f8ae902dc.png"
#         # with open(test_image_path, "rb") as f:
#         #     test_image_data = f.read()
#         #
#         # result_buf = BytesIO(test_image_data)
#
#         result_buf = wait_for_result(
#             content=content_file,
#             style=style_file,
#             prompt="",
#             preprocessor="Contour"
#         )
#
#         if not result_buf:
#             return Response({"error": "Image generation failed or timed out."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#
#         # 결과를 다시 base64로 인코딩
#         img_base64 = base64.b64encode(result_buf.getvalue()).decode('utf-8')
#         result_buf.close()
#
#         return Response({
#             "image_base64": img_base64,
#             "mime_type": "image/png"
#         })
#
#     except Exception as e:
#         traceback.print_exc()
#         return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def image_upload_view(request):
    try:
        style_url = request.data.get("style")
        content_url = request.data.get("content")
        if not style_url or not content_url:
            return Response({"error": "Style and content image URLs are required."}, status=status.HTTP_400_BAD_REQUEST)

        # 이미지 URL로부터 데이터 가져오기
        style_response = requests.get(style_url)
        content_response = requests.get(content_url)

        if style_response.status_code != 200 or content_response.status_code != 200:
            return Response({"error": "Failed to download one or both images."}, status=status.HTTP_400_BAD_REQUEST)

        style_file = BytesIO(style_response.content)
        content_file = BytesIO(content_response.content)

        result_buf = wait_for_result(
            content=content_file,
            style=style_file,
            prompt="",
            preprocessor="Contour"
        )

        if not result_buf:
            return Response({"error": "Image generation failed or timed out."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 결과 이미지를 base64로 인코딩
        try:
            img_base64 = base64.b64encode(result_buf.getvalue()).decode('utf-8')
            result_buf.close()

        except Exception as e:
            traceback.print_exc()
            return Response({"error": f"Base64 encode failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "image_base64": img_base64,
            "mime_type": "image/png"
        })

    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
