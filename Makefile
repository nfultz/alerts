.PHONY = install clean


install : news-lambda.zip
	aws lambda delete-function --function-name neal_news_lambda || true
	aws lambda create-function --function-name neal_news_lambda \
		                       --runtime python3.6 \
							   --handler lambda_handler \
							   --role arn:aws:iam::887983324737:role/neal_news_lambda_permission \
							   --zip-file fileb://./$<

%/ :
	pip3 install --system $* -t .

news-lambda.zip : mysession.py bs4/ soupsieve/
	zip -r $@ $?

clean :
	rm -rf */ news-lambda.zip
