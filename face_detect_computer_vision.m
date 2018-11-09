

unknown_image = imread('visionteam.jpg');
imshow(unknown_image);shg;

faceDetectorTool = vision.CascadeObjectDetector;
shapeInserterTool = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',[0 255 255]);

bbox = step(faceDetectorTool, unknown_image);

% Draw boxes around detected faces and display results
detected_faces_image = step(shapeInserterTool, unknown_image, int32(bbox));

imshow(detected_faces_image), title('Detected faces')

for i = 1 : size(bbox, 1)     
  rectangle('Position', bbox(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
end 

for i = 1 : size(bbox, 1) 
  cropped_image = imcrop(detected_faces_image, bbox(i, :)); 
  figure(3);
  subplot(6, 6, i);
  imshow(cropped_image); 
end
