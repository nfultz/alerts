.PHONY = manual install clean

export AWS_DEFAULT_REGION = us-east-1

manual :
	./neal_news.py Google_Alert_-_Daily_Digest_21.eml

install : news-lambda.zip
	aws lambda delete-function --function-name neal_news_lambda || true
	aws lambda create-function --function-name neal_news_lambda                                  \
	                           --runtime python3.6                                               \
	                           --handler neal_news.lambda_handler                                \
	                           --timeout 60                                                      \
	                           --role arn:aws:iam::887983324737:role/neal_news_lambda_permission \
	                           --zip-file fileb://./$<
	echo "Don't forget to update SES to use new lambda"

%/ :
	pip3 install --system $* -t .

news-lambda.zip : neal_news.py bs4/ soupsieve/
	zip -r $@ $?

clean :
	rm -rf */ news-lambda.zip
