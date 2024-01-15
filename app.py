import tensorflow as tf
model = tf.keras.models.load_model('LSTM_model.h5')
import numpy as np
from flask import Flask, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
x_acc_list = []
y_acc_list = []
z_acc_list = []

@app.route('/update_data',methods=['GET'])
def update_data():
    x_acc_data = float(request.args.get("x_acc",None))
    y_acc_data = float(request.args.get("y_acc",None))
    z_acc_data = float(request.args.get("z_acc",None))

    global x_acc_list,y_acc_list,z_acc_list
    # 把資料讀取近來
   
    x_acc_list.append((x_acc_data  + 20) / 40)
    y_acc_list.append((y_acc_data  + 20) / 40)
    z_acc_list.append((z_acc_data  + 20) / 40)
    # 若陣列長度大於128，要把最舊的刪掉，可以使用list.pop(0)函式
    if(len(x_acc_list) > 128):
    	x_acc_list.pop(0)
    if(len(y_acc_list) > 128):
    	y_acc_list.pop(0)
    if(len(z_acc_list) > 128):
    	z_acc_list.pop(0)
    
    return 'ok'

@app.route('/get_data',methods=['GET'])
def get_data():
    # 當收到request時，預測目前坐姿，然後把結果回傳(return)
    
    # your code
    segments = []
    for i in range(128):
    	segments.append([x_acc_list[i], y_acc_list[i], z_acc_list[i]])
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 128, 3)
    predictions = model.predict(reshaped_segments)
    max_predictions = np.argmax(predictions, axis=1)
    class_labels =  ['Downstairs','Jogging','Sitting','Standing','Upstairs','Walking']
    return class_labels[max_predictions[0]]

app.run(host="0.0.0.0", port=3000, debug=False)