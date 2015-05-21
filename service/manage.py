from flask.ext.script import Manager
from flask.ext.migrate import Migrate, MigrateCommand
from models import db
from app import create_app

app = create_app()
migrate = Migrate(app, db)
manager = Manager(app)


class CreateDatabase(Command):
    """
    Creates the database based on models.py
    """

    def run(self):
        with create_app().app_context():
            db.create_all()


manager.add_command('db', MigrateCommand)
manager.add_command('createdb', CreateDatabase())

if __name__ == '__main__':
    manager.run()
