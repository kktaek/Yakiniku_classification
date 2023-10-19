import base64
import cv2 as cv
import numpy as np
from flask import jsonify, Flask, request, render_template, Response
from flask_cors import CORS
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

app = Flask(__name__)
CORS(app, supports_credentials=True)
model = None

def base64_to_img(base64_str):
    # RGB 형식의 base64가 입력으로 주어지면 RGB 형식의 numpy로 출력
    byte_data = base64.b64decode(base64_str)  # base64를 바이너리로 변환
    encode_image = np.asarray(bytearray(byte_data), dtype="uint8")  # 1차원 배열로 이진 변환
    img_array = cv.imdecode(encode_image, cv.IMREAD_COLOR)  # cv2 활용 3채널 매트릭스로 디코딩
    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)  # BGR2RGB
    return img_array


def img_to_base64(img_array):
    # RGB 형식의 numpy가 입력으로 주어지면 RGB 형식의 base64로 출력
    img_array = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)  # RGB2BGR，cv2 인코딩
    encode_image = cv.imencode(".jpg", img_array)[1]  # 1차원 배열로 변환
    byte_data = encode_image.tobytes()  # 바이너리 변환
    base64_str = base64.b64encode(byte_data).decode("ascii")  # base64 변환
    return base64_str


@app.route('/')
def index_html():
    return render_template('realtime_camera.html')


@app.route('/get_img', methods=['POST'])
def receive_pic():
    # receive the base64 image
    img_base64 = request.form.get('img')[len("data:image/png;base64,"):]

    # transform the base64 image to normal image
    img = base64_to_img(img_base64)

    # use the model to detect the image, input: `result`<numpy img(RGB)>, output: `result`<numpy img(RGB)>
#     model_path = "../../work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/epoch_500.pth"
#     config_path = "../../work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py"
#     model = init_detector(config_path, model_path, device='cuda:0')

    result = inference_detector(model, img)
    
    img_result = model.show_result(img, 
                                   result, 
                                   score_thr=0.3, 
                                   show=False, 
                                   wait_time=0, 
                                   win_name='result', 
                                   bbox_color=None, 
                                   text_color=(200, 200, 200), 
                                   mask_color=None, 
                                   out_file=None)
    print("Image shape: ", img.shape, "[origin], ", img_result.shape, "[detected]")
    # return the detected image
    base64_img = img_to_base64(img_result)

    respose = {
        "code": 200,
        "base64_img": "data:image/png;base64," + str(base64_img)
    }
    # compare
    # print("origin:", img_base64[0:50])
    # print("return:", base64_img[0:50])

    return jsonify(respose)


if __name__ == "__main__":
    model_path = "../../work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/epoch_500.pth"
    config_path = "../../work_dirs/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py"
    model = init_detector(config_path, model_path, device='cuda:0')
    app.run(host="0.0.0.0", port=5006, debug=False, ssl_context='adhoc')