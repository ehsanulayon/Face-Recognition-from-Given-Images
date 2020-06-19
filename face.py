import face_recognition 
import os
import cv2

Known_Faces = 'known_faces'
Unknown_Faces = 'unknown_faces'
Tolerance = 0.6
Frame_Thickness = 3
Font_Thickness = 2
Model = 'hog'

print('Loading Known Faces')

known_faces = []
known_names = []

for name in os.listdir(Known_Faces):
	for filename in os.listdir(f"{Known_Faces}/{name}"):
		image = face_recognition.load_image_file(f"{Known_Faces}/{name}/{filename}")
		encoding = face_recognition.face_encodings(image)[0]
		known_faces.append(encoding)
		known_names.append(name)

print("Processing Unknown Faces")

for filename in os.listdir(Unknown_Faces):
	print(filename)
	image = face_recognition.load_image_file(f"{Unknown_Faces}/{filename}")
	locations = face_recognition.face_locations(image, model = Model)
	encodings = face_recognition.face_encodings(image, locations)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	for face_encoding , face_location in zip (encodings,locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, Tolerance)
		match = None
		if True in results:
			match = known_names[results.index(True)] 
			print(f"Match Found: {match}")

			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])

			color = [255, 0 , 0]

			cv2.rectangle(image, top_left, bottom_right, color, Frame_Thickness)

			top_left = (face_location[3], face_location[2])
			bottom_right = (face_location[1], face_location[2]+22)

			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), Font_Thickness)
	cv2.imshow(filename, image)
	cv2.waitKey(10000)
	#cv2.destroyWindow(filename)		


