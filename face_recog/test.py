import face_recognition
import os
import cv2

known_dir= 'known_faces'
unknown_dir = 'unknown_faces'

knownFaces = []     #face encodings stored here
knownNames = []     #name of faces

for name in os.listdir(known_dir):      #encoding known faces
    for file in os.listdir(f'{known_dir}/{name}'):
        image = face_recognition.load_image_file(f'{known_dir}/{name}/{file}')
        encoding = face_recognition.face_encodings(image)[0]
        knownFaces.append(encoding)
        knownNames.append(name)

#creating encoding for unknown faces and recognizing them
for file in os.listdir(unknown_dir):
    image = face_recognition.load_image_file(os.path.join(unknown_dir, file))
    locations = face_recognition.face_locations(image, model = 'cnn')
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      #becoz opencv uses bgr format

    for face_encoding, face_location in zip(encodings, locations):
        result = face_recognition.compare_faces(knownFaces, face_encoding)
        if True in result:
            match = knownNames[result.index(True)]
            topLeft = (face_location[3], face_location[0])
            bottomRight = (face_location[1], face_location[2])
            cv2.rectangle(image, topLeft, bottomRight, (0,255,0), 2)        #rectangle for face

            topLeft = (face_location[3], face_location[2])
            bottomRight = (face_location[1], face_location[2]+20)
            cv2.rectangle(image, topLeft, bottomRight, (0,255,0), cv2.FILLED)

            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250,0,0), 1)
    cv2.imshow(file, image)
    cv2.waitKey(0)
    cv2.destroyWindow(file)


