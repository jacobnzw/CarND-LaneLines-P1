# **Finding Lane Lines on the Road** 

**Goals**

The goal of this project is to design an image processing pipeline, which will detect road lanes from a video stream taken by a front-mounted camera on a car.

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[hough]: ./examples/result_step_3.jpg
[final]: ./examples/result_step_4.jpg


---

### Pipeline Description

The steps in my pipeline are the following:

1. Convert image to grayscale
2. Apply Gaussian bluring with
3. Canny edge detection
4. Apply a mask to select the region of interest
5. Hough transform for finding lines
6. Filter out lines with infinite and out-of-range derivatives
7. Fit a line
8. Draw the lines

First, I convert the image to grayscale. Then I apply Gaussian blurring with `kernel_size=3`, which doesn't really make much of a difference in the end result, but I keep it there anyway in case the image gets too sharp. In the following step, I apply the Canny edge detection algorithm with parameters `low_threshold=120` and `high_threshold=240`. Then I apply the region of interest mask in order to focus the attention on the road, where the lanes are. Designing the polygonal ROI was a trial and error process with many iterations, for which I had to write a piece of code to help me visualize the mask boundaries. 

The next step applies the Hough transform for detecting lines in the image. I used the following values of Hough transform parameters:

* `rho=1`
* `theta=1*np.pi/180`
* `threshold=15`
* `min_line_len=50`
* `max_line_gap=60`

The transform produces the following output
![Hough transform][hough]

I'm showing this particular frame, because it was a stumbling block in the pipeline development for some time.

In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function by adding computation of the derivatives, which I use to filter out the lines with infinite derivatives. I further assumed, that the right and left lane lines will have approximately the same absolute value of the derivative. For one image, I plotted a histogram of the line derivatives and they appear to be clustered around $ \pm 1.5 $. This is why I'm filtering out lines for which

$$ 1 < \frac{y_2 - y_1}{x_2 - x_1} < 2. $$

This little heuristic ensures that the spurious horizontal lines are removed from consideration. Next I took the few remaining line coordinates and I fitted the first-degree polynomial using the `polyfit()` function and finally the `polyval()` function for evaluating the fitted line for drawing.

The end result looks like this
![Final result][final]


### 2. Identify potential shortcomings with your current pipeline

One shortcoming I see is the algorithm's reliance on the range of derivatives the lines must have. The designed pipeline would work well on straight roads or on long drawn out curves, like those on a highway. For sharper bends it would fail.

Another potential shortcoming could be limited robustness to changing weather conditions, for which the pipeline has not been tested.


### 3. Suggest possible improvements to your pipeline

A possible improvement for sharper bends would be to use a more flexible model for the lanes. For example, using a second-degree polynomial instead of a line might give better results. Of course one would have to use either a more general Hough transform (some extensions beyond lines and circles exist) or use the ordinary one, but then one would have to again divide the detected line segments into left/right lane and then fit a more flexible model to those points.
