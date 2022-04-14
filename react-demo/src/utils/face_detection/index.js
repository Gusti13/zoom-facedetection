// import face api
import * as faceapi from 'face-api.js'

export default async function runDetection(VIDEO, DETECT_CANVAS, width, height) {
    let ctx = DETECT_CANVAS.getContext("2d");
    
    async function detec(){
    //   console.log(faceapi.nets);
      const detections = await faceapi.detectAllFaces(VIDEO)
                                      .withFaceLandmarks()
                                      .withFaceExpressions()
                                      .withAgeAndGender();
      
       const json = detections.map(val => {
           const json = {
               age: val.age,
               expressions: val.expressions,
               gender: {
                   gender: val.gender,
                   genderProbability: val.genderProbability
               }
           }
           return json
       })

    //    console.log(json)


      // hapus canvas yg sebelumnya
      ctx.clearRect(0,0, width, height);
      // buat canvas baru
      const resizedDetections = faceapi.resizeResults(detections, { width, height})
      faceapi.draw.drawDetections(DETECT_CANVAS, resizedDetections);
      faceapi.draw.drawFaceLandmarks(DETECT_CANVAS, resizedDetections);
      faceapi.draw.drawFaceExpressions(DETECT_CANVAS, resizedDetections);
    }


    Promise.all([
      faceapi.nets.ageGenderNet.loadFromUri('/models'),
      faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
      faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
      faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
      faceapi.nets.faceExpressionNet.loadFromUri('/models')
    ]).then(() => {
      setInterval(detec, 100)
    });
  
}