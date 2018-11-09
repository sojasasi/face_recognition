%https://in.mathworks.com/matlabcentral/fileexchange/45750-face-recognition-using-pca
%http://openbio.sourceforge.net/resources/eigenfaces/eigenfaces-html/facesOptions.html

function preprocessingImage()

	datapath = '';
	D = dir(datapath);
	
	%calculate no of images in a directory
	imgcount = 0;
	for i=1 : size(D,1)
		if not(strcmp(D(i).name,'Thumbs.db')
			imgcount = imgcount + 1;
		end
	end
	
	%Preprocessing
	preprocessed_image_matrix = [];
	for i = 1 : imgcount
		str = strcat(datapath,'\','tface_',int2str(i),'.jpg');
		img = imread(str);

		%1 Converting RGB scale to grayscale
		img = rgb2gray(img);
		
		%resize the image to standardize image size
		img = imresize(img, [500,500]);
		
		%Reshaping the matrix
		[row col] = size(img);
		temp = reshape(img',row*col,1); %the transpose of image is taken because of reshape function
		preprocessed_image_matrix = [preprocessed_image_matrix temp];
	end

	mean_face = mean(preprocessed_image_matrix, 2); %mean value of a particular pixel of training images
	normalised_image_matrix = preprocessed_image_matrix - repmat(mean_face, [1, imgcount]);
	
	covariance_matrix = [];
	covariance_matrix = normalised_image_matrix' * normalised_image_matrix;
	[eigen_vector, diag_eigen_matrix] = eig(covariance_matrix);
	
	largest_eigenvectors = [];
	for i = 1 : size(eigen_vector,2) 
    	if( diag_eigen_matrix(i,i) > 1 ) %Kaisers rule
        	largest_eigenvectors = [largest_eigenvectors eigen_vector(:,i)];
    	end
	end
	
	eigen_faces = normalised_image_matrix * largest_eigenvectors;
	
	%calculate the weight space for each known individual by projecting their faceimages onto facespaces
	weights_known = [ ];  % projected image vector matrix
	for i = 1 : size(eigenfaces,2)
		temp = eigenfaces' * normalised_image_matrix(:,i);
		weights_known = [weights_known temp];
	end

	testDir = '';
	unknown_face_image = imread(testDir);
	unknown_face_image = unknown_face_image(:,:,1);
	[r c] = size(unknown_face_image);
	temp = reshape(unknown_face_image',r*c,1);
	temp = double(temp)- mean_face; % for this the size of the unknown image and known image needs to be same.. figure out
	
	%calculate weights of the unknown input images with the eigenfaces
	weights_unknown = eigenfaces' * temp;
	
	euclide_dist = [];
	for i=1 : size(eigenfaces,2)
		temp = (norm(weights_unknown - weights_known(:,i)))^2;
		euclide_dist = [euclide_dist temp];
	end
	[euclide_dist_min recognized_index] = min(euclide_dist);
	recognized_img = strcat('tface_',int2str(recognized_index),'.jpg');

end
