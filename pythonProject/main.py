import werkzeug
import flask
from flask import Flask
from flask_restful import Api, Resource
import ImageCaptionModel


class ImageCaptionGenerator(Resource):
    @staticmethod
    def get():
        return "Image Uploaded Successfully"

    @staticmethod
    def post():

        image_file = flask.request.files['image']
        filename = werkzeug.utils.secure_filename(image_file.filename)
        image_file.save(filename)
        result = ImageCaptionModel.predict_caption('androidFlask.jpg', 'greedy')
        result = ImageCaptionModel.clean_caption(result)
        return ImageCaptionModel.caption_preprocessed(result)


app = Flask(__name__)
api = Api(app)
api.add_resource(ImageCaptionGenerator, '/')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
