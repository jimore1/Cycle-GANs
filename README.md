# Cycle GANs

>   GANs(Generative Adversarial Networks),GANs are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. They are used widely in image generation, video generation and voice generation.

#### 1.Two main compositions

    To understand GANs, you should know how generative algorithms work, and for that, contrasting them with discriminative algorithms is instructive. Discriminative algorithms try to classify input data; that is, given the features of an instance of data, they predict a label or category to which that data belongs.

    For example, given all the words in an email (the data instance), a discriminative algorithm could predict whether the message is `spam` or `not_spam`. `spam` is one of the labels, and the bag of words gathered from the email are the features that constitute the input data. When this problem is expressed mathematically, the label is called `y` and the features are called `x`. The formulation `p(y|x)` is used to mean “the probability of y given x”, which in this case would translate to “the probability that an email is spam given the words it contains.”

* Generative Algorithms

>     So discriminative algorithms map features to labels. They are concerned solely with that correlation. One way to think about generative algorithms is that they do the opposite. Instead of predicting a label given certain features, they attempt to predict features given a certain label.

* Discriminative Algorithms

>     The question a generative algorithm tries to answer is: Assuming this email is spam, how likely are these features? While discriminative models care about the relation between `y` and `x`, generative models care about “how you get x.” They allow you to capture `p(x|y)`, the probability of `x` given `y`, or the probability of features given a label or category. (That said, generative algorithms can also be used as classifiers. It just so happens that they can do more than categorize input data.)

#### 2.The working process

1. In one GANs,there are two neural networks.One neural network, called the `generator`, generates new data instances, while the other, the `discriminator`, evaluates them for authenticity; i.e. the discriminator decides whether each instance of data that it reviews belongs to the actual training dataset or not.
  
2. Let’s say we’re trying to do something more banal than mimic the Mona Lisa. We’re going to generate hand-written numerals like those found in the MNIST dataset, which is taken from the real world. The goal of the discriminator, when shown an instance from the true MNIST dataset, is to recognize those that are authentic.
  
3. Meanwhile, the generator is creating new, synthetic images that it passes to the discriminator. It does so in the hopes that they, too, will be deemed authentic, even though they are fake. The goal of the generator is to generate passable hand-written digits: to lie without being caught. The goal of the discriminator is to identify images coming from the generator as fake.
  

###### <font color="green">Here are the steps a GAN takes:</font>

* The generator takes in random numbers and returns an image.
* This generated image is fed into the discriminator alongside a stream of images taken from the actual, ground-truth dataset.
* <font color="red">The discriminator takes in both real and fake images and returns probabilities, a number between 0 and 1, with 1 representing a prediction of authenticity and 0 representing fake.</font>

###### <font color="blue">So you have a double feedback loop:</font>

* The discriminator is in a feedback loop with the ground truth of the images, which we know.
* The generator is in a feedback loop with the discriminator.

![GAN schema](https://wiki.pathmind.com/images/wiki/gan_schema.png)

###### <font color="brown">There are 2 generators (G and F) and 2 discriminators (X and Y) being trained here.</font>

* Generator `G` learns to transform image `X` to image `Y`. 
* Generator `F` learns to transform image `Y` to image `X`. 
* Discriminator `D_X` learns to differentiate between image `X` and generated image `X` (`F(Y)`).
* Discriminator `D_Y` learns to differentiate between image `Y` and generated image `Y` (`G(X)`).

![Cyclegan model](https://tensorflow.google.cn/static/tutorials/generative/images/cyclegan_model.png)

#### 3.Loss function

>     Cycle consistency means the result should be close to the original input. For example, if one translates a sentence from English to French, and then translates it back from French to English, then the resulting sentence should be the same as the original sentence.

In cycle consistency loss,

* Image  $X$ is passed via generator $G$ that yields generated image $\hat{Y}$ .
* Generated image $\hat{Y}$ is passed via generator $F$ that yields cycled image $\hat{X}$ .
* Mean absolute error is calculated between $X$ and $\hat{X}$ .

            $forward$ $cycle$ $consistency$ $loss$ $:$ $X->G(X)->F(G(X))$ ~ $\hat{X}$

            $backward$ $cycle$ $consistency$ $loss$ $:$ $Y->F(Y)->G(F(Y))$ ~ $\hat{Y}$            

![Cycle loss](https://tensorflow.google.cn/static/tutorials/generative/images/cycle_loss.png)

As shown above, generator $G$ is responsible for translating image $X$ to image $Y$ . Identity loss says that, if you fed image $Y$ to generator $G$ , it should yield the real image $Y$ or something close to image $Y$ .

If you run the zebra-to-horse model on a horse or the horse-to-zebra model on a zebra, it should not modify the image much since the image already contains the target class.

                                                                $Identity$ $loss$ $=$ $|G(Y)-Y|+|F(X)-X|$

#### 4.training process

    @tf.function
    def train_step(real_x, real_y):
      # persistent is set to True because the tape is used more than
      # once to calculate the gradients.
      with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
    
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)
    
        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
    
        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)
    
        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
    
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)
    
        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
    
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
    
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    
      # Calculate the gradients for generator and discriminator
      generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                            generator_g.trainable_variables)
      generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                            generator_f.trainable_variables)
    
      discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                discriminator_x.trainable_variables)
      discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                discriminator_y.trainable_variables)
    
      # Apply the gradients to the optimizer
      generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                generator_g.trainable_variables))
    
      generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                generator_f.trainable_variables))
    
      discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                    discriminator_x.trainable_variables))
    
      discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                    discriminator_y.trainable_variables))
