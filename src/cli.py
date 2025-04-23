
#!/usr/bin/env python3
"""
Command line interface for the Prompt Optimization Platform
"""
import click
import os
from alembic.config import Config
from alembic import command
import sys
from pathlib import Path

# Get base directory  
BASE_DIR = Path(__file__).resolve().parent.parent

@click.group()
def cli():
    """Command line interface for prompt optimizer database management"""
    pass

@cli.command()
def init_db():
    """Initialize the database and run migrations"""
    from scripts.init_database import init_database, create_default_settings
    
    # Create data directory if it doesn't exist
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Initialize database
    if init_database():
        create_default_settings()
        click.echo("Database initialized and default settings created")
    else:
        click.echo("Failed to initialize database")

@cli.command()
@click.argument("message")
def create_migration(message):
    """Create a new migration"""
    alembic_cfg = Config(str(BASE_DIR / "alembic.ini"))
    command.revision(alembic_cfg, autogenerate=True, message=message)
    click.echo(f"Migration created with message: {message}")

@cli.command()
def upgrade_db():
    """Upgrade database to latest migration"""
    alembic_cfg = Config(str(BASE_DIR / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")
    click.echo("Database upgraded to latest migration")

@cli.command()
@click.argument("revision", default="head")
def downgrade_db(revision):
    """Downgrade database to a specific revision"""
    alembic_cfg = Config(str(BASE_DIR / "alembic.ini"))
    command.downgrade(alembic_cfg, revision)
    click.echo(f"Database downgraded to revision: {revision}")

@cli.command()
def show_migrations():
    """Show migration history"""
    alembic_cfg = Config(str(BASE_DIR / "alembic.ini"))
    command.history(alembic_cfg)

@cli.command()
def backup_db():
    """Backup the SQLite database"""
    import datetime
    import shutil
    
    db_path = BASE_DIR / "data" / "prompt_optimizer.db"
    backup_dir = BASE_DIR / "data" / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"prompt_optimizer_{timestamp}.db"
    
    if db_path.exists():
        shutil.copy2(db_path, backup_path)
        click.echo(f"Database backed up to {backup_path}")
    else:
        click.echo("Database file not found")

@cli.command()
@click.argument("backup_file")
def restore_db(backup_file):
    """Restore the SQLite database from a backup"""
    import shutil
    
    db_path = BASE_DIR / "data" / "prompt_optimizer.db"
    backup_path = Path(backup_file)
    
    if not backup_path.exists():
        click.echo(f"Backup file not found: {backup_file}")
        return
    
    if db_path.exists():
        click.confirm("Existing database will be overwritten. Continue?", abort=True)
    
    shutil.copy2(backup_path, db_path)
    click.echo(f"Database restored from {backup_file}")

if __name__ == "__main__":
    cli()
