%https://in.mathworks.com/matlabcentral/fileexchange/45750-face-recognition-using-pca
%http://openbio.sourceforge.net/resources/eigenfaces/eigenfaces-html/facesOptions.html

function preprocessingImage()

	datatpath = '';
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

		%Reshaping the matrix
		[row col] = size(img);
		temp = reshape(img',row*col,1); 
		preprocessed_image_matrix = [preprocessed_image_matrix temp];
	end

	normalised_image_matrix = preprocessed_image_matrix - mean(preprocessed_image_matrix);
	
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
	
	testDir = '';
	unknown_face_image = imread(testDir);
	unknown_face_image = unknown_face_image(:,:,1);
	[r c] = size(unknown_face_image);
	temp = reshape(unknown_face_image',r*c,1);
	temp = double(temp)- mean(unknown_face_image);
	eigenface_of_unknown_image = eigenfaces' * temp;
	
	euclide_dist = [ ];
	for i=1 : size(eigenfaces,2)
		temp = (norm(projtestimg-projectimg(:,i)))^2;
		euclide_dist = [euclide_dist temp];
	end
	[euclide_dist_min recognized_index] = min(euclide_dist);
	recognized_img = strcat(int2str(recognized_index),'.jpg');

end