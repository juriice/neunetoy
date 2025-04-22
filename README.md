# Neunetoy
This is my personal code space for experimenting with Neural Networks while exercising Rust programming.

## What to do with this code
You can use it along with your favorite Neural Network book or article to experiment and sense their capabilities
and limitations.

If you are alreday here, you have probably realized that there are already few other Rust implementations and many
other Python implementations that does the same. Here I try to model and write Neural Networks code in a simple and
understandable manner with the goal to understand their characteristics. In any case if your goal is to learn them
the best is to write them on your own ;).

## How-to run this code
If you are not already doing Rust, first is to get familiar with it and at least install Rust [tool-chain](https://doc.rust-lang.org/book/ch01-00-getting-started.html).

Clone the code:
```bash
git clone https://github.com/juriice/neunetoy.git
cd neunetoy
```

Run the given examples:
```bash
time cargo run --release
```

To sense the need for processor power and clever (or non clever) code optimizations technics do the next exercise
and compare the execution time with the previous run:
```bash
time cargo run
```
