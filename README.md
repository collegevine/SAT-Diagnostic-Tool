# SAT-Diagnostic-Tool

## Deployment URL
https://sat-diagnostic-tool.herokuapp.com/    

## Run
$ export FLASK_APP=app.py     
$ flask run

## Deploy (Quick)
$ heroku login    
$ heroku git:remote -a sat-diagnostic-tool    
$ git push heroku master    

## Deploy (Fresh)
$ heroku login    
$ cd ~ && mkdir deploy-SAT    
$ git clone git@github.com:collegevine/SAT-Diagnostic-Tool.git    
$ cd SAT-Diagnostic-Tool    
$ virtual venv    
$ source venv/bin/activate    
$ pip install -r requirements.txt    
$ pip install gunicorn    
$ pip freeze > requirements.txt    
$ echo "web: gunicorn app:app" > Procfile    
$ heroku git:remote -a sat-diagnostic-tool    
$ git add requirements.txt Procfile     
$ git commit -m "prep for deploy"    
$ git push origin master    

