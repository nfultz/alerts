.PHONY = manual  update install clean docker favicon

export AWS_DEFAULT_REGION = us-east-1

manual :
	./neal_news.py Google_Alert_-_Daily_Digest_21.eml

update : news-lambda.zip
	aws lambda update-function-code --function-name neal_news_lambda                             \
	                                --zip-file fileb://./$<

install : news-lambda.zip
	aws lambda delete-function --function-name neal_news_lambda || true
	aws lambda create-function --function-name neal_news_lambda                                  \
	                           --runtime python3.8                                               \
	                           --handler neal_news.lambda_handler                                \
	                           --timeout 60                                                      \
	                           --role arn:aws:iam::887983324737:role/neal_news_lambda_permission \
	                           --zip-file fileb://./$<
	echo "Don't forget to update SES to use new lambda"

%/ :
	pip3 install $* -t .

news-lambda.zip : neal_news.py bs4/ soupsieve/
	zip -r $@ $?

clean :
	rm -rf */ news-lambda.zip

docker : Dockerfile entry.sh analysis.py download-model.py gen_features.py neal_news.py score.py train1.py train.py
	$$(aws ecr get-login --no-include-email --region us-east-1) && \
	docker build --squash -t neal-news . && \
	docker tag neal-news:latest 887983324737.dkr.ecr.us-east-1.amazonaws.com/neal-news:latest && \
	docker push 887983324737.dkr.ecr.us-east-1.amazonaws.com/neal-news:latest



favicon :
	aws s3 cp favicon.ico s3://www.neal.news/favicon.ico
