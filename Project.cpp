#include "mainwindow.h"
#include <math.h>
#include "ui_mainwindow.h"
#include <QtGui>
#include <ctime>

/***********************************************************************
  This is the only file you need to change for your assignment.  The
  other files control the UI (in case you want to make changes.)
************************************************************************/


// The first four functions provide example code to help get you started

// Convert an image to grey-scale
void MainWindow::BlackWhiteImage(QImage *image)
{
    int r, c;
    QRgb pixel;

    for(r=0;r<image->height();r++)
    {
        for(c=0;c<image->width();c++)
        {
            pixel = image->pixel(c, r);
            double red = (double) qRed(pixel);
            double green = (double) qGreen(pixel);
            double blue = (double) qBlue(pixel);

            // Compute intensity from colors - these are common weights
            double intensity = 0.3*red + 0.6*green + 0.1*blue;

            image->setPixel(c, r, qRgb( (int) intensity, (int) intensity, (int) intensity));
        }
    }
}

// Add random noise to the image
void MainWindow::AddNoise(QImage *image, double mag, bool colorNoise)
{
    int r, c;
    QRgb pixel;
    int noiseMag = mag;
    noiseMag *= 2;

    for(r=0;r<image->height();r++)
    {
        for(c=0;c<image->width();c++)
        {
            pixel = image->pixel(c, r);
            int red = qRed(pixel);
            int green = qGreen(pixel);
            int blue = qBlue(pixel);

            // If colorNoise, add color independently to each channel
            if(colorNoise)
            {
                red += rand()%noiseMag - noiseMag/2;
                green += rand()%noiseMag - noiseMag/2;
                blue += rand()%noiseMag - noiseMag/2;
            }
            // otherwise add the same amount of noise to each channel
            else
            {
                int noise = rand()%noiseMag - noiseMag/2;

                red += noise;
                green += noise;
                blue += noise;
            }

            // Make sure we don't over or under saturate
            red = min(255, max(0, red));
            green = min(255, max(0, green));
            blue = min(255, max(0, blue));

            image->setPixel(c, r, qRgb( red, green, blue));
        }
    }
}

// Here is an example of blurring an image using a mean or box filter with the specified radius.
// This could be implemented using separable filters to make it much more efficient, but it is not.
void MainWindow::MeanBlurImage(QImage *image, int radius)
{
    if(radius == 0)
        return;

    int r, c, rd, cd, i;
    QRgb pixel;

    // This is the size of the kernel
    int size = 2*radius + 1;

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int w = image->width();
    int h = image->height();

    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute kernel to convolve with the image.
    double *kernel = new double [size*size];

    for(i=0;i<size*size;i++)
    {
        kernel[i] = 1.0;
    }

    // Make sure kernel sums to 1
    double denom = 0.000001;
    for(i=0;i<size*size;i++)
        denom += kernel[i];
    for(i=0;i<size*size;i++)
        kernel[i] /= denom;

    // For each pixel in the image...
    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double rgb[3];

            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     pixel = buffer.pixel(c + cd + radius, r + rd + radius);

                     // Get the value of the kernel
                     double weight = kernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }

            // Store mean pixel in the image to be returned.
            image->setPixel(c, r, qRgb((int) floor(rgb[0] + 0.5), (int) floor(rgb[1] + 0.5), (int) floor(rgb[2] + 0.5)));
        }
    }

    // Clean up.
   delete [] kernel;
}

// Downsample the image by 1/2
void MainWindow::HalfImage(QImage &image)
{
    QImage buffer;
    int w = image.width();
    int h = image.height();
    int r, c;

    buffer = image.copy();

    // Reduce the image size.
    image = QImage(w/2, h/2, QImage::Format_RGB32);

    // Copy every other pixel
    for(r=0;r<h/2;r++)
        for(c=0;c<w/2;c++)
        {
             image.setPixel(c, r, buffer.pixel(c*2, r*2));
        }
}




void MainWindow::GaussianBlurImage(QImage *image, double sigma)
{

	int start_s=clock(); 
     //Add your code here.  Look at MeanBlurImage to get yourself started.
    double rr, s = 2.0 * sigma * sigma;
    // sum is for normalization --sum should =1
    double sum = 0.000001;
	int rd, cd, i,j;
	//radius=sigma*sqrt(2*log(255))-1;
	int radius= 5*sigma-1;
	int size = 2*radius+1;

    QRgb pixel;
    // Create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int w = image->width();
    int h = image->height();
	
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);
	double *gKernel = new double [size*size];

	for(i=0;i<size;i++)
	{
		for (j=0;j<size;j++)
		{
		    rr = (i-radius)*(i-radius)+(j-radius)*(j-radius);
            gKernel[j+i*size] = (exp(-(rr)/s))/(M_PI * s);  
		}
		}
			for(int i = 0; i < size*size;i++)
            sum += gKernel[i];
            for(int i = 0; i <size*size;i++)
            gKernel[i] /= sum;
//for each pixel in the image
	for(int y=0;y<h;y++) 
	{
        for(int x=0;x<w;x++)
        {
            double rgb[3];
            
            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;
            // Convolve the kernel at each pixel
			for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     pixel = buffer.pixel(x + cd + radius, y + rd + radius);

                     // Get the value of the kernel
                     double weight = gKernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }
         image->setPixel(x, y, qRgb((int) floor(rgb[0]+0.5), (int) floor(rgb[1]+0.5), (int) floor(rgb[2]+0.5)));	
		}				 			
      }
int stop_s=clock();
qDebug("Running time:%f seconds",double(stop_s-start_s)/double(CLOCKS_PER_SEC));
}

void MainWindow::SeparableGaussianBlurImage(QImage *image, double sigma)
{
    // Add your code here.  Done right, you should be able to copy most of the code from GaussianBlurImage.
	int start_s=clock(); 
	double rc,rr, s = 2.0 * sigma * sigma;
    // sum is for normalization --sum should =1
    double sum = 0.000001;
	int rd, cd, i,j;
	//radius=sigma*sqrt(2*log(255))-1;
	int radius= 5*sigma-1;
	int const size = 2*radius+1;
    QRgb pixel;
    // Create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int w = image->width();
    int h = image->height();

    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);
	//define two separated matrix: cKernel is size by 1 matrix, rKernel is 1 by size matrix: 
	double *cKernel = new double [size];
	//double *rKernel = new double [size];

	for(i=0;i<size;i++)
	{
		rc = (i-radius)*(i-radius);
        cKernel[i] = (exp(-(rc)/s))/(M_PI * s);        
	}
		for(int i = 0; i < size;i++)
        sum += cKernel[i];

        for(i=0;i<size;i++)
        cKernel[i] /= sum;
    

  //for(j=0;j<size;j++)
  //     {		 
		//    rr = (j-radius)*(j-radius);
  //          rKernel[j] = (exp(-(rr)/s))/(M_PI * s);         
		//}
	 //for(int i = 0; i < size;i++)
	 //sum += rKernel[i];
	 //for(i=0;i<size;i++)
  //   rKernel[i] /= sum;

//for each pixel in the image
	for(int y=0;y<h;y++) 
	{
        for(int x=0;x<w;x++)
        {
            double rgb[3];
            
            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;
            // Convolve the kernel at each pixel
			for(cd=-radius;cd<=radius;cd++)
			{
			// Get the pixel value
			pixel = buffer.pixel(x + cd + radius, y + radius);
			// Get the value of the kernel
			double weight = cKernel[cd + radius];
			rgb[0] += weight*(double) qRed(pixel);
			rgb[1] += weight*(double) qGreen(pixel);
			rgb[2] += weight*(double) qBlue(pixel);
		    }
         image->setPixel(x, y, qRgb((int) floor(rgb[0]+0.5), (int) floor(rgb[1]+0.5), (int) floor(rgb[2]+0.5)));	
	    //convlove the resulted matrix with row matrix: 
		}
	}

//for each pixel in the image
	for(int y=0;y<h;y++) 
	{
        for(int x=0;x<w;x++)
        {
            double rgb[3];
            
            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;
            // Convolve the kernel at each pixel
			for(rd=-radius;rd<=radius;rd++)
			{
				// Get the pixel value
				pixel = buffer.pixel(x + radius, y + rd + radius);
				// Get the value of the kernel
				double weight = cKernel[rd + radius];
				rgb[0] += weight*(double) qRed(pixel);
				rgb[1] += weight*(double) qGreen(pixel);
				rgb[2] += weight*(double) qBlue(pixel);
			}
			image->setPixel(x, y, qRgb((int) floor(rgb[0]+0.5), (int) floor(rgb[1]+0.5), (int) floor(rgb[2]+0.5)));	
		}
	}
	int stop_s=clock();
    qDebug("Running time:%f seconds",double(stop_s-start_s)/double(CLOCKS_PER_SEC));
}



void MainWindow::FirstDerivImage(QImage *image, double sigma)
{
    // Add your code here.
    double rr, s = 2.0 * sigma * sigma;
    // sum is for normalization --sum should =1
    double sum = 0.000001;
	int rd, cd, i,j;
	//radius=sigma*sqrt(2*log(255))-1;
	int radius= 5*sigma-1;
	int size = 2*radius+1;
	QRgb pixel;
    QRgb current,next;

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int w = image->width();
    int h = image->height();
	
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);
	double *gKernel = new double [size*size];
		for(i=0;i<size;i++)
	
		for (j=0;j<size;j++)
		{
		rr = (i-radius)*(i-radius)+(j-radius)*(j-radius);
        gKernel[i] = (exp(-(rr)/s))/(M_PI * s); 
		}
	
		for(int i = 0; i < size;i++)
        sum += gKernel[i];
        for(i=0;i<size;i++)
        gKernel[i] /= sum;

//calculate the first derivative of x axis:

	for(int y=0;y<h;y++) 
	{
        for(int x=0;x<w;x++)
        {
            double rgb[3];
            
            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;
            // Convolve the kernel at each pixel
			for(rd=-radius;rd<=radius;rd++)
			{
				// Get the pixel value
				pixel = buffer.pixel(x + radius, y + rd + radius);
				// Get the value of the kernel
				double weight = gKernel[rd + radius];
				rgb[0] += weight*(double) qRed(pixel);
				rgb[1] += weight*(double) qGreen(pixel);
				rgb[2] += weight*(double) qBlue(pixel);
			}
			image->setPixel(x, y, qRgb((int) floor(rgb[0]+0.5), (int) floor(rgb[1]+0.5), (int) floor(rgb[2]+0.5)));	
		}
	}
			QImage Deriv_1;
			Deriv_1=image->copy(0,0,w+1,h+1);

for(int y=0;y<h;y++) 
	{
        for(int x=0;x<w;x++)
        {
            double rgb[3];
            // Get the pixel value
			//the two neibouring pixel on x-axis:
            current = Deriv_1.pixel(x,y);
			next=Deriv_1.pixel(x+1,y);
			//QRgb next= Deriv_1.pixel(x+1,y);

			rgb[0] = (double)qRed(next) - (double)qRed(current) + 128;
            rgb[1] = (double)qGreen(next) - (double)qGreen(current) + 128;
            rgb[2] = (double)qBlue(next) - (double)qBlue(current) + 128;

			//Make sure we don't over or under saturate
			for (int k=0;k<3;k++){
                if(rgb[k] > 255) rgb[k] = 255;
                if(rgb[k] < 0) rgb[k] = 0;
			}
                
         image->setPixel(x, y, qRgb((int) floor(rgb[0]+0.5), (int) floor(rgb[1]+0.5), (int) floor(rgb[2]+0.5)));	
		}				 			
      }

}

void MainWindow::SecondDerivImage(QImage *image, double sigma)
{
    // Add your code here.
	int rd, cd, i,j;
	QRgb pixel;

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int w = image->width();
    int h = image->height();
	
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(0,0, w, h);
	SeparableGaussianBlurImage(image,sigma);

	for(int y=0;y<h;y++) 
	   {
        for(int x=0;x<w;x++)
        {
            QRgb current=buffer.pixel(x,y);
	        QRgb next=image->pixel(x,y);
			double rgb[3];

			rgb[0] = (double)qRed(next) - (double)qRed(current) + 128;
            rgb[1] = (double)qGreen(next) - (double)qGreen(current) + 128;
            rgb[2] = (double)qBlue(next) - (double)qBlue(current) + 128;

			//Make sure we don't over or under saturate
			for (int k=0;k<3;k++){
                if(rgb[k] > 255) rgb[k] = 255;
                if(rgb[k] < 0) rgb[k] = 0;
			}
                
         image->setPixel(x, y, qRgb((int) floor(rgb[0]+0.5), (int) floor(rgb[1]+0.5), (int) floor(rgb[2]+0.5)));	
		}				 			
      }
}

void MainWindow::SharpenImage(QImage *image, double sigma, double alpha)
{
    // Add your code here.  It's probably easiest to call SecondDerivImage as a helper function.
	//sharpen Image is just use the origin image - alpha*(image convolve second derivative Gaussian)
	
	int w = image->width();
    int h = image->height();
	QImage buffer = image->copy(0,0,w,h);
	SecondDerivImage(&buffer,sigma);

	for(int y=0;y<h;y++) 
	{
        for(int x=0;x<w;x++)
        {
            QRgb current=buffer.pixel(x,y);
	        QRgb next=image->pixel(x,y);
			double rgb[3];

			rgb[0] = (double)qRed(next) - alpha*((double)qRed(current) - 128);
            rgb[1] = (double)qGreen(next) -alpha* ((double)qGreen(current) - 128);
            rgb[2] = (double)qBlue(next) -alpha* ((double)qBlue(current) - 128);

			//Make sure we don't over or under saturate
			for (int k=0;k<3;k++)
			{
                if(rgb[k] > 255) rgb[k] = 255;
                if(rgb[k] < 0) rgb[k] = 0;
             }
			           
         image->setPixel(x, y, qRgb((int) floor(rgb[0]+0.5), (int) floor(rgb[1]+0.5), (int) floor(rgb[2]+0.5)));	
		}				 			
      }
}

void MainWindow::BilateralImage(QImage *image, double sigmaS, double sigmaI)
{
    // Add your code here.  Should be similar to GaussianBlurImage.
	double rr, s = 2.0 * sigmaS * sigmaS;
    // sum is for normalization --sum should =1
    double sum = 0.000001;
	int x, y, rd, cd, i,j;
	//radius=sigma*sqrt(2*log(255))-1;
	int radius= 5*sigmaS-1;
	int size = 2*radius+1;
	QRgb pixel;
    QRgb center;

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int w = image->width();
    int h = image->height();
	
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);
	
	double *gKernel = new double [size*size];
		for(i=0;i<size;i++)
	
		for (j=0;j<size;j++)
		{
		rr = (i-radius)*(i-radius)+(j-radius)*(j-radius);
        gKernel[i] = (exp(-(rr)/s))/(M_PI * s); 
		}
	
		for(int i = 0; i < size;i++)
        sum += gKernel[i];
        for(i=0;i<size;i++)
        gKernel[i] /= sum;

//calculate the first derivative of x axis:

	for(int y=0;y<h;y++) 
	{
        for(int x=0;x<w;x++)
        {
            double rgb[3];
			double RGB[3];            
            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;

			RGB[0] = 0.0;
            RGB[1] = 0.0;
            RGB[2] = 0.0;

            // Convolve the kernel at each pixel
			for(rd=-radius;rd<=radius;rd++)
				for (cd=-radius;cd<=radius;cd++)
				{
				// Get the pixel value
				 pixel = buffer.pixel(x + cd + radius, y + rd + radius);
				 center = buffer.pixel(x,y);
				// Get the value of the kernel
				 double weight = gKernel[rd + radius];

				 RGB[0] += weight*exp(-((double) qRed(pixel)-(double) qRed(center))*((double) qRed(pixel)-(double) qRed(center))/(2*sigmaI*sigmaI));
                 RGB[1] += weight*exp(-((double) qGreen(pixel)-(double) qGreen(center))*((double) qGreen(pixel)-(double) qGreen(center))/(2*sigmaI*sigmaI));
                 RGB[2] += weight*exp(-((double) qBlue(pixel)-(double) qBlue(center))*((double) qBlue(pixel)-(double) qBlue(center))/(2*sigmaI*sigmaI));
 
 
                 rgb[0] += weight*(double) qRed(pixel)*exp(-((double) qRed(pixel)-(double) qRed(center))*((double) qRed(pixel)-(double) qRed(center))/(2*sigmaI*sigmaI));
                 rgb[1] += weight*(double) qGreen(pixel)*exp(-((double) qGreen(pixel)-(double) qGreen(center))*((double) qGreen(pixel)-(double) qGreen(center))/(2*sigmaI*sigmaI));
                 rgb[2] += weight*(double) qBlue(pixel)*exp(-((double) qBlue(pixel)-(double) qBlue(center))*((double) qBlue(pixel)-(double) qBlue(center))/(2*sigmaI*sigmaI));
			     }
			     rgb[0]=rgb[0]/RGB[0];
                 rgb[1]=rgb[1]/RGB[1];
                 rgb[2]=rgb[2]/RGB[2];
				 image->setPixel(x, y, qRgb((int) floor(rgb[0]+0.5), (int) floor(rgb[1]+0.5), (int) floor(rgb[2]+0.5)));	
		  }
	}
}

void MainWindow::SobelImage(QImage *image)
{
    // Add your code here.

	int i,j;
	int rd, cd;
	QRgb pixel;
	//horizontal: 
	double sobely[3][3]={1, 2, 1, 0, 0, 0, -1, -2, -1};
	//vertical
	double sobelx[3][3]={1, 0, -1, 2, 0, -2, 1, 0, -1};

	BlackWhiteImage(image); 
	int h=image->height();
	int w=image->width();
	
	QImage buffer = image->copy(-1,-1,w+2,h+2);
    
	for(int y=0;y<h;y++)
    {
        for(int x=0;x<w;x++)           
        {
            
			double grayx = 0.0;
            double grayy = 0.0;
			for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                {
                  pixel = buffer.pixel(x+cd+1,y+rd+1);
				  double weighty = sobely[rd+1][cd+1];
				  double weightx = sobelx[rd+1][cd+1];
				  grayy += weighty*(double) qGray(pixel);
                  grayx += weightx*(double) qGray(pixel);
				}
            //magnitude of sobel is: g=sqrt(gx^2+gy^2)
			//orientation: theta=tan-1(gy/gx)
			double mag=sqrt(grayy*grayy+grayx*grayx); // magnitude of the gradient
			double orien= atan2 (grayy,grayx); // orientation of the gradient
			double red = (sin(orien) + 1.0)/2.0;
    		double green = (cos(orien) + 1.0)/2.0;
    		double blue = 1.0 - red - green;

    		red *= mag*4.0;
    		green *= mag*4.0;
    		blue *= mag*4.0;
			
    		// Make sure the pixel values range from 0 to 255
    		red = min(255.0, max(0.0, red));
    		green = min(255.0, max(0.0, green));
    		blue = min(255.0, max(0.0, blue));
            image->setPixel(x, y, qRgb( red, green, blue));
       }

	}

}


void MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{
// Add your code here.  Return the RGB values for the pixel at location (x,y) in double rgb[3]. 
    QRgb Q11,Q12,Q21,Q22;

	int h=image->height();
	int w=image->width();

	int x1=(int)floor(x);
    int y1=(int)floor(y);

    int x2=(int)floor(x)+1;
	int y2=(int)floor(y)+1;
 
    QImage buffer = image->copy(0,0,w+2,h+2);

    if(x1>=0&&x1<w&&y1>=0&&y1<h)
        Q11=buffer.pixel(x1,y1);
    else
        Q11= 0;
    if(x1>=0&&x1<w&&y2>=0&&y2<h)
        Q12=buffer.pixel(x1,y2);
    else
        Q12= 0;
    if(x2>=0&&x2<w&&y1>=0&&y1<h)
        Q21=buffer.pixel(x2,y1);
    else
        Q21= 0;
    if(x2>=0&&x2<w&&y2>=0&&y2<h)
        Q22=buffer.pixel(x2,y2);
    else
        Q22= 0;

    rgb[0]=((x2-x)*(y2-y)*(double)qRed(Q11)+(x-x1)*(y2-y)*(double)qRed(Q21)+(x2-x)*(y-y1)*(double)qRed(Q12)+(x-x1)*(y-y1)*(double)qRed(Q22));
    rgb[1]=((x2-x)*(y2-y)*(double)qGreen(Q11)+(x-x1)*(y2-y)*(double)qGreen(Q21)+(x2-x)*(y-y1)*(double)qGreen(Q12)+(x-x1)*(y-y1)*(double)qGreen(Q22));
    rgb[2]=((x2-x)*(y2-y)*(double)qBlue(Q11)+(x-x1)*(y2-y)*(double)qBlue(Q21)+(x2-x)*(y-y1)*(double)qBlue(Q12)+(x-x1)*(y-y1)*(double)qBlue(Q22));

}


// Here is some sample code for rotating an image.  I assume orien is in degrees.

void MainWindow::RotateImage(QImage *image, double orien)
{
    int r, c;
    QRgb pixel;
    QImage buffer;
    int w = image->width();
    int h = image->height();
    double radians = -2.0*3.141*orien/360.0;

    buffer = image->copy();

    pixel = qRgb(0, 0, 0);
    image->fill(pixel);

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double rgb[3];
            double x0, y0;
            double x1, y1;

            // Rotate around the center of the image.
            x0 = (double) (c - w/2);
            y0 = (double) (r - h/2);

            // Rotate using rotation matrix
            x1 = x0*cos(radians) - y0*sin(radians);
            y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (w/2);
            y1 += (double) (h/2);

            BilinearInterpolation(&buffer, x1, y1, rgb);

            image->setPixel(c, r, qRgb((int) floor(rgb[0] + 0.5), (int) floor(rgb[1] + 0.5), (int) floor(rgb[2] + 0.5)));
        }
    }

}

void MainWindow::FindPeaksImage(QImage *image, double thres)
{
 //   // Add your code here.
	int i,j;
	int rd, cd;
	QRgb pixel;
 
	//horizontal: 
	double sobely[3][3]={1, 2, 1, 0, 0, 0, -1, -2, -1};
	//vertical
	double sobelx[3][3]={1, 0, -1, 2, 0, -2, 1, 0, -1};
	int h=image->height();
	int w=image->width();
	
	QImage buffer;
	BlackWhiteImage(image); 
	
	buffer=image->copy(-1,-1,w+2,h+2);

	double **magn=new double*[w];
	for(int i=0;i<w;i++) magn[i]=new double[h];

    double **gray_x=new double*[w];
	for(int i=0;i<w;i++) gray_x[i]=new double[h];

	double **gray_y=new double*[w];
	for(int i=0;i<w;i++) gray_y[i]=new double[h];

    double Q11,Q12,Q21,Q22; 

	//QImage buffer = image->copy(-1,-1,w+2,h+2);
 
	for(int r=0;r<h;r++)
      for(int c=0;c<w;c++)           
        {
 			double grayx = 0.0;
            double grayy = 0.0;  
			for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                {
                  pixel = buffer.pixel(c+cd+1,r+rd+1);

				  double weighty = sobely[rd+1][cd+1];
				  double weightx = sobelx[rd+1][cd+1];

				  grayy += weighty*(double) qGray(pixel);
                  grayx += weightx*(double) qGray(pixel);
				}

            //magnitude of sobel is: g=sqrt(gx^2+gy^2)
			magn[c][r]=sqrt(grayy*grayy+grayx*grayx); // magnitude of the gradient
			gray_y[c][r]= grayy;
		    gray_x[c][r]= grayx;
			}

		for(int r=0;r<h;r++)
            for(int c=0;c<w;c++)
            {

			   double x0=gray_x[c][r]/magn[c][r];
			   double y0=gray_y[c][r]/magn[c][r];
			   
			   double t0=c+x0;
			   double m0=r+y0;
			   double t1=c-x0;
			   double m1=r-y0;

			   int x1=(int)floor(t0);
               int y1=(int)floor(m0);
               int x2=(int)floor(t1)+1;
	           int y2=(int)floor(m1)+1;			   


			   double **image1=magn; 
			   if(x1>=0&&x1<w&&y1>=0&&y1<h)
				   Q11=image1[x1][y1];
			   else
				   Q11= 0;
			   if(x1>=0&&x1<w&&y2>=0&&y2<h)
				   Q12=image1[x1][y2];
			   else
				   Q12= 0;
			   if(x2>=0&&x2<w&&y1>=0&&y1<h)
				   Q21=image1[x2][y1];
			   else
				   Q21= 0;
			   if(x2>=0&&x2<w&&y2>=0&&y2<h)
				   Q22=image1[x2][y2];
			   else
				   Q22= 0;

			   double e0=((x2-t0)*(y2-m0)*Q11+(t0-x1)*(y2-m0)*Q21+(x2-t0)*(m0-y1)*Q12+(t0-x1)*(m0-y1)*Q22);
			   double e1=((x2-t1)*(y2-m1)*Q11+(t1-x1)*(y2-m1)*Q21+(x2-t1)*(m1-y1)*Q12+(t1-x1)*(m1-y1)*Q22);	
			   //A pixel is a peak response if it is larger than the threshold ("thres"), e0, and e1. 
	           //Assign the peak responses a value of 255 and everything else 0.
	           if (magn[c][r]>thres && magn[c][r]>e0 && magn[c][r]>e1 )
	           image->setPixel(c,r,qRgb(255,255,255));
	           else
	           image->setPixel(c,r,qRgb(0,0,0));
			}
	}


void MainWindow::MedianImage(QImage *image, int radius)
{
    // Add your code here
}

void MainWindow::HoughImage(QImage *image)
{
    // Add your code here
}

void MainWindow::CrazyImage(QImage *image)
{
    int i,j;
	int rd, cd;
	QRgb pixel;
	//horizontal: 
	double sobely[3][3]={1, 2, 1, 0, 0, 0, -1, -2, -1};
	//vertical
	double sobelx[3][3]={1, 0, -1, 2, 0, -2, 1, 0, -1};

	BlackWhiteImage(image); 
	int h=image->height();
	int w=image->width();
	
	QImage buffer = image->copy(-1,-1,w+2,h+2);
    
	for(int y=0;y<h;y++)
    {
        for(int x=0;x<w;x++)           
        {
            
			double grayx = 0.0;
            double grayy = 0.0;
			for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                {
                  pixel = buffer.pixel(x+cd+1,y+rd+1);
				  double weighty = sobely[rd+1][cd+1];
				  double weightx = sobelx[rd+1][cd+1];
				  grayy += weighty*(double) qGray(pixel);
                  grayx += weightx*(double) qGray(pixel);
				}
            //magnitude of sobel is: g=sqrt(gx^2+gy^2)
			//orientation: theta=tan-1(gy/gx)
			double mag=sqrt(grayy*grayy+grayx*grayx); // magnitude of the gradient
			double orien= atan2 (grayy,grayx); // orientation of the gradient
			
			if (mag>10)
			mag=10;
			if (orien>0.5*M_PI)
				orien=0;

			double red = (sin(orien) + 1.0)/2.0;
    		double green = (cos(orien) + 1.0)/2.0;
    		double blue = 1.0 - red - green;

    		red *= mag*4.0;
    		green *= mag*4.0;
    		blue *= mag*4.0;
			
    		// Make sure the pixel values range from 0 to 255
    		red = min(255.0, max(0.0, red));
    		green = min(255.0, max(0.0, green));
    		blue = min(255.0, max(0.0, blue));
            image->setPixel(x, y, qRgb( red, green, blue));
       }

	}
}

