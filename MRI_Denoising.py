import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from matplotlib import patches
import plotly.express as px
from tqdm import tqdm

import math

class MRI_Denoising(object):

    def __init__(self):
        """ Initialization of object
        """
        pass

    def view_img(self,img,figsize=(15, 15)):
        """ View image with matplotlib

        Args:
            img (np.array) : 2D numpy array of image
            figsize (tuple): figure size
        """
        figure(figsize=figsize)
        plt.imshow(img,cmap='gray')
        plt.show()


    def SNR(self,img):
        """ Return signal-to-noise ratio of an MRI image

        Args:
            img (np.array) : 2D numpy array of MRI image

        Returns:
            signal-to-noise ratio (float)

        """
        background_std=np.std(self.get_img_background(img))
        signal_mean=np.mean(self.get_img_signal_area(img))
        return signal_mean/background_std


    def get_img_background(self,img):
        """ Return background image of an MRI image

        Args:
            img (np.array) : 2D numpy array of MRI image

        Returns:
            background (np.array) : 2D numpy array of background image

        """
        return self.search_window(img,(0,0),20)



    def get_img_signal_area(self,img):
        """ Return signal area image of an MRI image

        Args:
            img (np.array) : 2D numpy array of MRI image

        Returns:
            signal area (np.array) : 2D numpy array of signal area image

        """
        return self.search_window(img,(140,65),10)

    def show_img_background(self,img,figsize=(15, 15),linewidth=2):
        """ Shows image with the background area colored in red

        Args:
            img (np.array) : 2D numpy array of MRI image
            figsize (tuple) : size of figure
            linewidth (int): size of line
        """
        fig,ax=plt.subplots(figsize=figsize)
        rect=patches.Rectangle((0,0),20,20,linewidth=linewidth,edgecolor='r',facecolor='none')

        ax.imshow(img,cmap='gray')
        ax.add_patch(rect)
        plt.show()


    def show_img_signal_area(self,img,linewidth=2):
        """ Shows image with the signal area colored in red

        Args:
            img (np.array) : 2D numpy array of MRI image
            figsize (tuple) : size of figure
            linewidth (int): size of line
        """
        fig,ax=plt.subplots(figsize=figsize)
        rect=patches.Rectangle((65,140),10,10,linewidth=linewidth,edgecolor='r',facecolor='none')

        ax.imshow(img,cmap='gray')
        ax.add_patch(rect)
        plt.show()

    def show_img_background_and_signal(self,img,figsize=(15, 15),linewidth=2):
        """ Shows image with background area colored in red and signal area colored in yellow

        Args:
            img (np.array) : 2D numpy array of MRI image
            figsize (tuple) : size of figure
            linewidth (int): size of line
        """
        fig,ax=plt.subplots(figsize=figsize)
        rect_background=patches.Rectangle((0,0),20,20,linewidth=linewidth,edgecolor='r',facecolor='none')
        rect_signal=patches.Rectangle((65,140),10,10,linewidth=linewidth,edgecolor='y',facecolor='none')

        ax.imshow(img,cmap='gray')
        ax.add_patch(rect_background)
        ax.add_patch(rect_signal)
        plt.show()


    def plot_background_histogram(self,img_noisy,img_denoised,facet=False):
        """ Plots histogram of pixel values for background for 2 images, noisy and denoised

        Args:
            img_noisy (np.array) : 2D numpy array of noisy MRI image
            img_denoised (np.array) : 2D numpy array of denoised MRI image

        """
        noisy_background=self.get_img_background(img_noisy).ravel()
        denoised_background=self.get_img_background(img_denoised).ravel()

        n=['Noisy']*len(noisy_background)
        dn=['Denoised']*len(denoised_background)

        histogram={
            'Pixel Values':list(noisy_background)+list(denoised_background),
            'image':n+dn
        }

        histogram=pd.DataFrame(histogram)

        if facet==True:
            fig = px.histogram(histogram,x='Pixel Values',color='image',facet_col='image',color_discrete_sequence=px.colors.qualitative.D3,title='Backroung Image Histogram')
        else:
            fig = px.histogram(histogram,x='Pixel Values',color='image',color_discrete_sequence=px.colors.qualitative.D3,title='Backroung Image Histogram')

        fig.show()


    def plot_signal_histogram(self,img_noisy,img_denoised,facet=False):
        """ Plots histogram of pixel values for signal area for 2 images, noisy and denoised

        Args:
            img_noisy (np.array) : 2D numpy array of noisy MRI image
            img_denoised (np.array) : 2D numpy array of denoised MRI image

        """
        noisy_signal=self.get_img_signal_area(img_noisy).ravel()
        denoised_signal=self.get_img_signal_area(img_denoised).ravel()

        n=['Noisy']*len(noisy_signal)
        dn=['Denoised']*len(denoised_signal)

        histogram={
            'Pixel Values':list(noisy_signal)+list(denoised_signal),
            'image':n+dn
        }

        histogram=pd.DataFrame(histogram)

        if facet==True:
            fig = px.histogram(histogram,x='Pixel Values',color='image',facet_col='image',color_discrete_sequence=px.colors.qualitative.D3,title='Signal Histogram')
        else:
            fig = px.histogram(histogram,x='Pixel Values',color='image',color_discrete_sequence=px.colors.qualitative.D3,title='Signal Histogram')

        fig.show()


    def search_window(self,img,p,radius):
        """ Returns a square neighborhood centered at pixel p (x,y) with specified radius

        Args:
            img (np.array) : 2D numpy array of noisy MRI image
            p (tuple): pixel location
            radius (int): radius of search window

        Returns:
            img (np.array) : 2D numpy array of background image

        """

        row=p[0]
        col=p[1]

        min_boundary=0
        max_boundary_col=(len(img[0])-1)
        max_boundary_row=(len(img)-1)

        # row window
        from_row=row-math.floor(radius/2)
        to_row=row+math.ceil(radius/2)


        if from_row<0:
            to_row=to_row+abs(from_row)
            from_row=0
        elif to_row>max_boundary_row:
            from_row=from_row-(to_row-max_boundary_row)
            to_row=max_boundary_row


        # column window
        from_col=col-math.floor(radius/2)
        to_col=col+math.ceil(radius/2)

        if from_col<0:
            to_col=to_col+abs(from_col)
            from_col=0
        elif to_col>max_boundary_col:
            from_col=from_col-(to_col-max_boundary_col)
            to_col=max_boundary_col


        return img[from_row:to_row,
                   from_col:to_col]


    def iterate_pixels(self,img):
        """ Returns a list of tuples of pixel locations for an image

        Args:
            img (np.array) : 2D numpy array of noisy MRI image

        Returns:
            pixels (list): list of pixel locations

        """
        pixels=[]
        for i in range(len(img)):
            for j in range(len(img[0])):
                pixels.append((i,j))
        return pixels

    def filter_pixel(self,img,p,Rsim,h):
        """ Filters a pixel with NLM formula

        Args:
            img (np.array) : Search window of MRI image
            p (tuple): pixel to be filtered
            Rsim (int): radius of local search window
            h (float): exponential decay function to control smoothing

        Returns:
            filtered_pixel (float): filtered pixel value

        """
        pixels=self.iterate_pixels(img)
        return np.sum(np.array(list(map(lambda q: img[q[0],q[1]]*self.w(img,p,q,Rsim,h), pixels))))

    def Z(self,img,p,Rsim,h):
        """ Normalization constant

        Args:
            img (np.array) : Search window of MRI image
            p (tuple): pixel to be filtered
            Rsim (int): radius of local search window
            h (float): exponential decay function to control smoothing

        Returns:
            Z (float): normalization constant

        """
        pixels=self.iterate_pixels(img)
        return np.sum(np.array(list(map(lambda q: (np.exp(-self.d(img,p,q,Rsim)/h**2)), pixels))))

    def w(self,img,p,q,Rsim,h):
        """ Weight associated between pixel p and pixel q

        Args:
            img (np.array) : Search window of MRI image
            p (tuple): pixel to be filtered
            Rsim (int): radius of local search window
            h (float): exponential decay function to control smoothing

        Returns:
            w (float): similarity weight for p and q

        """
        return (1/self.Z(img,p,Rsim,h))*np.exp(-self.d(img,p,q,Rsim)/h**2)

    def d(self,img,p,q,Rsim,sigma=1):
        """ Distance (similarity) associated between pixel p and pixel q

        Args:
            img (np.array) : Search window of MRI image
            p (tuple): pixel to be filtered
            Rsim (int): radius of local search window
            h (float): exponential decay function to control smoothing
            sigma (float)=1: sigma value for gaussian kernel

        Returns:
            d (float): Distance(similarity) scire

        """
        if (p,q) in self.memoization_d: return self.memoization_d[(p,q)]
        else:
            Np=self.search_window(img,p,Rsim)
            Nq=self.search_window(img,q,Rsim)
            self.memoization_d[(p,q)]=(np.linalg.norm(Np-Nq)**2/(2.*sigma**2))
            return self.memoization_d[(p,q)]


    def unpad(self,img, pad):
        """ Removes padding from image

        Args:
            img (np.array) : 2D numpy array of mage
            pad (tuple): padding specification

        Returns:
            unapped img (np.array): 2D numpy array with padding removed

        """
        nx, ny  = img.shape

        pad_x=pad[0]
        pad_y=pad[1]
        return img[pad_x:nx-pad_y,
                pad_x:ny-pad_y]


    def estimate_noise(self,img):
        """ Estimating noise from background image

        Args:
            img (np.array) : 2D numpy array of image

        Returns:
            noise (float): estimated noise

        """
        return np.sqrt(np.mean(self.get_img_background(img))/2)

    def NLM(self,img,Rsearch,Rsim,h,padding=(0,0)):
        """ Non-Local Mean denoising

        Args:
            img (np.array) : 2D numpy array of image
            Rsearch (int): radius of the search window
            Rsim (int): radius of patch to compare
            h (float): exponential decay value
            padding (tuple): padding to be added to the image

        Returns:
            denoised image (np.array): image denoised with NLM

        """

        # padding the image
        if padding!=(0,0):img=np.pad(img, padding)

        # memoization for saving distance values for pixel (p,q):d(p,q)
        self.memoization_d={}

        # initializing empty image of same shape
        filtered_img=np.zeros(img.shape)

        # iterate through each pixel in (x,y) direction for 2D image
        for i in tqdm(range(len(img))):
            for j in range(len(img[0])):

                # get pixel value
                p=(i,j)

                # obtain filtered pixel value for pixel p
                filtered_img[i,j]=self.filter_pixel(self.search_window(img,p,Rsearch),p,Rsim,h)

        # return filtered image with removed padding
        return self.unpad(filtered_img,(padding))

    def UNLM(self,img,Rsearch,Rsim,h,padding=(0,0)):
        """ Unbiased Non-Local Mean denoising

        Args:
            img (np.array) : 2D numpy array of image
            Rsearch (int): radius of the search window
            Rsim (int): radius of patch to compare
            h (float): exponential decay value
            padding (tuple): padding to be added to the image

        Returns:
            denoised image (np.array): image denoised with UNLM

        """
        std=self.estimate_noise(img)

        return np.sqrt(np.power(self.NLM(img,Rsearch,Rsim,h,padding),2)-2*std**2)
