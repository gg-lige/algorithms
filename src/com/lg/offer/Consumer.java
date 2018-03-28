package com.lg.offer;

import java.util.concurrent.BlockingQueue;

public class Consumer implements Runnable {
	BlockingQueue<String> queue;
	
	public Consumer(BlockingQueue<String> queue) {
		// TODO Auto-generated constructor stub
		this.queue=queue;
	}
	
	@Override
	public void run() {
		// TODO Auto-generated method stub
		try {
			String temp =queue.take();//如果队列为空，会阻塞当前线程
			System.out.println(temp);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
