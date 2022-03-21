use std::sync::{mpsc, Arc, Mutex};
use std::thread;

pub struct ThreadPool<T: Send + 'static, S: Send + 'static> {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Message<T, S>>,
    receiver: mpsc::Receiver<T>,
}

type Job<T, S> = Box<dyn FnOnce(&mut S) -> Option<T> + Send + 'static>;
enum Message<T: Send + 'static, S: Send + 'static> {
    Job(Job<T, S>),
    Terminate,
}

impl<T: Send + 'static, S: Send + 'static> ThreadPool<T, S> {
    pub fn new<F: FnMut() -> S>(size: usize, mut state_f: F) -> Self {
        let (m_sender, m_receiver) = mpsc::channel();
        let m_receiver = Arc::new(Mutex::new(m_receiver));

        let (res_sender, res_receiver) = mpsc::channel();

        let workers = (0..size)
            .map(|_| Worker::new(Arc::clone(&m_receiver), res_sender.clone(), state_f()))
            .collect();

        Self {
            workers,
            sender: m_sender,
            receiver: res_receiver,
        }
    }

    pub fn execute<F: FnOnce(&mut S) -> Option<T> + Send + 'static>(&self, f: F) {
        let job = Box::new(f);
        self.sender.send(Message::Job(job)).unwrap();
    }

    pub fn results(&self, count: usize) -> impl Iterator<Item = T> + '_ {
        (0..count).map(|_| self.receiver.recv().unwrap())
    }
}

impl<T: Send + 'static, S: Send + 'static> Drop for ThreadPool<T, S> {
    fn drop(&mut self) {
        for _ in &self.workers {
            self.sender.send(Message::Terminate).unwrap();
        }

        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

struct Worker {
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new<T: Send + 'static, S: Send + 'static>(
        receiver: Arc<Mutex<mpsc::Receiver<Message<T, S>>>>,
        sender: mpsc::Sender<T>,
        state: S,
    ) -> Self {
        let thread = thread::spawn(move || {
            let mut state = state;
            loop {
                let message = receiver.lock().unwrap().recv().unwrap();

                match message {
                    Message::Job(job) => {
                        if let Some(res) = job(&mut state) {
                            sender.send(res).unwrap();
                        }
                    }
                    Message::Terminate => break,
                }
            }
        });

        Self {
            thread: Some(thread),
        }
    }
}
