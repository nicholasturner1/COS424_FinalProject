import Nick as n
import main as m

print "Importing Data..."
X, y = n.import_file('train.csv')
cvx, cvy = n.import_file('cv.csv')
testx, testy = n.import_file('test.csv')

for n_splits in (2, 4, 6, 8, 10):

	model = n.train_Melody(X, y, n_splits=n_splits)

	preds_cv = n.predict_model(model, cvx, cvy)
	preds_test = n.predict_model(model, testx, testy)

	m.score_preds(preds_cv)

	m.save_preds(preds_cv, 'Melody_%d_year_cv_preds.csv' % n_estimators)
	m.save_preds(preds_test, 'Melody_%d_year_test_preds.csv' % n_estimators)