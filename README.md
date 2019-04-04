# KNN-Handwriting-Recognition
Program uses K-Nearest Neighbours algorithm to read handwritten input letter

Main feature of this program is that it is based on CUDA technology. Vast majority of the calculations are done parallelly on graphic card which makes calculation time much shorter.

How it works?
First we take input image and transform it to binary vector. After that we compare it with every image of the dataset so we can determine similarity. This part is done on a graphic card so in fact we compare our input with 64 (default value) dataset images at once. All results are then stored in list of results. When computing is finished this list is sorted and 100 best matches makes our KNN. Letter with the biggest number of appearance in KNN is the winner.

Sounds good, doesn't work.

So what went wrong?
First of all, program is terribly slow and there are two reasons for that. First is that GPU calculations are faster than CPU ones but only when we don't take memory allocation into consideration. 64 images at once are not enough to make it all faster in total. Okay so why don't we take 128 images at once, right? Of course it can be done and quite easly in fact since I tried to make that program as resizable as I could. That would most probably do the trick but there is reason number two which unfortunately made that idea pointless. Reason number two is that reading all the dataset files takes more than 4 minutes and there is not much we can do about it. This fact actually killed the whole idea of this program since it's unacceptable to wait that long.

Conclusions:
- In my opinion KNN algorithm is definitely not the right way to recognise handwriting. It makes us go through whole dataset (which has to be big when we take letters into consideration) every time. It is not efficient at all. On the other hand we can't deny that it is simple to implement and gives pretty good results.
- Parallel computing is a great tool but we need have an idea how to use it correctly and if there is a point to use it at all.

Even though the final outcome is not great I'm still quite glad I've made it because it was in fact a good lesson in terms of different programming approach (parallel computing) and results of bad design assumptions.


