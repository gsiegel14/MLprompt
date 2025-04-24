
#!/usr/bin/env python
"""
Command line interface for the ML Prompt Optimization Platform
"""
import click
import os
from alembic.config import Config
from alembic import command
import sys
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Get base directory  
BASE_DIR = Path(__file__).resolve().parent.parent

@click.group()
def cli():
    """Command line interface for prompt optimizer database management"""
    pass

@cli.command()
def init_db():
    """Initialize the PostgreSQL database and run migrations"""
    # Parse database URL
    db_url = os.getenv("DATABASE_URL", "postgresql://promptopt:devpassword@localhost:5432/promptopt")
    parts = db_url.split("/")
    db_name = parts[-1]
    connection_str = "/".join(parts[:-1]) + "/postgres"
    
    try:
        # Connect to postgres database to create our database
        conn = psycopg2.connect(connection_str)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            click.echo(f"Creating database {db_name}")
            cursor.execute(f'CREATE DATABASE "{db_name}"')
        
        cursor.close()
        conn.close()
        
        # Run migrations
        alembic_cfg = Config(str(BASE_DIR / "alembic.ini"))
        command.upgrade(alembic_cfg, "head")
        click.echo("Database initialized and migrations applied")
        
    except Exception as e:
        click.echo(f"Error initializing database: {e}")
        sys.exit(1)

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
@click.option("--tag", help="Tag name for the backup")
def backup_db(tag):
    """Backup PostgreSQL database"""
    db_url = os.getenv("DATABASE_URL", "postgresql://promptopt:devpassword@localhost:5432/promptopt")
    parts = db_url.split("/")
    db_name = parts[-1]
    user_pass = parts[2].split("@")[0]
    username = user_pass.split(":")[0]
    host_port = parts[2].split("@")[1]
    host = host_port.split(":")[0]
    
    # Create backups directory if it doesn't exist
    backup_dir = BASE_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    # Create backup filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{tag}" if tag else ""
    backup_file = backup_dir / f"{db_name}_{timestamp}{tag_suffix}.sql"
    
    # Run pg_dump command
    command = f"pg_dump -h {host} -U {username} -d {db_name} -f {backup_file}"
    click.echo(f"Running backup command: {command}")
    
    if os.system(command) == 0:
        click.echo(f"Database backed up to {backup_file}")
    else:
        click.echo("Database backup failed")

@cli.command()
@click.argument("backup_file")
def restore_db(backup_file):
    """Restore PostgreSQL database from backup"""
    db_url = os.getenv("DATABASE_URL", "postgresql://promptopt:devpassword@localhost:5432/promptopt")
    parts = db_url.split("/")
    db_name = parts[-1]
    user_pass = parts[2].split("@")[0]
    username = user_pass.split(":")[0]
    host_port = parts[2].split("@")[1]
    host = host_port.split(":")[0]
    
    backup_path = Path(backup_file)
    if not backup_path.exists():
        backup_path = BASE_DIR / "backups" / backup_file
        if not backup_path.exists():
            click.echo(f"Backup file not found: {backup_file}")
            return
    
    # Run psql command
    command = f"psql -h {host} -U {username} -d {db_name} -f {backup_path}"
    click.echo(f"Running restore command: {command}")
    
    if os.system(command) == 0:
        click.echo(f"Database restored from {backup_path}")
    else:
        click.echo("Database restore failed")

if __name__ == "__main__":
    cli()
