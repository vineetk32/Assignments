package org.apache.hadoop.examples;

import java.io.IOException;
import java.lang.Math;
import java.lang.Long;
import java.util.ArrayList;
import java.lang.Integer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;

public class TwitterPopularity {
	
	// --------------------------------
	// Mapper class
	// --------------------------------
	
	public static class TweetClassifier extends Mapper<LongWritable, Text, Text, IntWritable>{

		// Map function
		public void map(LongWritable fileOffset, Text fileLine, Context context) throws IOException, InterruptedException {
		
			Text author = new Text();
			Text txttopic = new Text();
			IntWritable num = new IntWritable(1);

			
			// Split line into chunks
			String[] tweetChunks = fileLine.toString().split(",\"");
			
			// Find the author and tweet content
			String origAuthor = null;
			String tweetContent = null;
			for(String tweetChunk : tweetChunks){
				String chunkKey = tweetChunk.split("\":")[0].replace("\"", "");
				if((chunkKey.equals("screen_name")) && (origAuthor == null)){
					origAuthor = tweetChunk.split("\":")[1].replace("\"", "");
				}
				else if(chunkKey.equals("text") && (tweetContent == null)){
					tweetContent = tweetChunk.split("\":")[1].replace("\"", "");
				}
				if((origAuthor != null) && (tweetContent != null))
					break;
			}
			
			// Quit if we didn't get an author or content
			if((origAuthor == null) || (tweetContent == null))
				return;
			
			/* PA3: Add code here to emit intermediate result for per-author processing */
			author.set("@" + origAuthor);
			num.set(tweetContent.length());
			context.write(author,num);
			
			// Search the tweet content for #topics and save them, avoiding duplicates
			ArrayList<String> topicList = new ArrayList<String>(0);
			String[] contentChunks = tweetContent.split(" ");
			for(String contentChunk : contentChunks){
				if(contentChunk.length() > 0 && (contentChunk.charAt(0) == '#') && (!topicList.contains(removePunct(contentChunk)))){
					topicList.add("#" + removePunct(contentChunk));
				}
			}
			
			/* PA3: Add code here to emit intermediate result for per-topic processing */
			num.set(1);
			for ( String topic : topicList)
			{
				txttopic.set(topic);
				context.write(txttopic,num);
			}
		}
		
		// Punctuation remover
		private static String removePunct(String input){
			return input.replace(":", "").replace(".", "").replace(",", "").replace("?", "").replace("#", "").replace("!", "");
		}
		
	}
	
	// --------------------------------
	// Reducer class
	// --------------------------------
	
	public static class StatConsolidator extends Reducer<Text,IntWritable,Text,Text> {
	
		// Reduce function

		public void reduce(Text subject, Iterable<IntWritable> occurrenceLengths, Context context) throws IOException, InterruptedException {
			
			/* PA3: Add code here to carry out per-user and per-topic reduction 
				  and emit final output */

			int sum = 0;
			int numTweets = 0;
			for (IntWritable val : occurrenceLengths) {
				sum += val.get();
				numTweets++;
			}
			//We want avg length per tweet for a user. Re-using the vriable sum
    		Text result = new Text();
			if (subject.charAt(0) == '@')
			{
				sum = sum / numTweets;
			}
			result.set(Integer.toString(sum));
			context.write(subject,result);
		}
		
	}
	
	// --------------------------------
	// Main function
	// --------------------------------
	
	// Main function
	public static void main(String[] args) throws Exception {
		
		// Create config
		Configuration conf = new Configuration();
		
		// Parse arguments
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 2) {
			System.err.println("Usage: twitterpopularity <in> <out>");
			System.exit(2);
		}
		
		// Create new job with name
		Job job = new Job(conf, "TwitterPopularity");
		
		// Set jar
		job.setJarByClass(TwitterPopularity.class);
		
		// Set map and reduce classes
		job.setMapperClass(TweetClassifier.class);
		job.setReducerClass(StatConsolidator.class);
		
		// Set intermediate format classes
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
		
		// Set output format classes
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		// Set paths
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		
		// Set number of reducers
		job.setNumReduceTasks(6);
		
		// Wait for job to complete
		System.exit(job.waitForCompletion(true) ? 0 : 1);
		
	}
}
