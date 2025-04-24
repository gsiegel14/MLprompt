
"""create ml tables

Revision ID: 04b04a4ccb21
Revises: 
Create Date: 2025-04-24 11:44:14.892449

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime


# revision identifiers, used by Alembic.
revision = '04b04a4ccb21'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create model_configurations table
    op.create_table(
        'model_configurations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('primary_model', sa.String(255), nullable=False),
        sa.Column('optimizer_model', sa.String(255), nullable=False),
        sa.Column('temperature', sa.Float(), default=0.0),
        sa.Column('max_tokens', sa.Integer(), default=1024),
        sa.Column('top_p', sa.Float(), default=1.0),
        sa.Column('top_k', sa.Integer(), default=40),
        sa.Column('is_default', sa.Boolean(), default=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('user.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    )
    
    # Create metric_configurations table
    op.create_table(
        'metric_configurations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('metric_weights', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('target_threshold', sa.Float(), default=0.8),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('user.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    )
    
    # Create ml_experiments table
    op.create_table(
        'ml_experiments',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('model_config_id', sa.String(36), sa.ForeignKey('model_configurations.id'), nullable=True),
        sa.Column('metric_config_id', sa.String(36), sa.ForeignKey('metric_configurations.id'), nullable=True),
        sa.Column('status', sa.String(50), default="created"),
        sa.Column('result_data', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('user.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    )
    
    # Create ml_experiment_iterations table
    op.create_table(
        'ml_experiment_iterations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('experiment_id', sa.String(36), sa.ForeignKey('ml_experiments.id'), nullable=False),
        sa.Column('iteration_number', sa.Integer(), nullable=False),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('output_prompt', sa.Text(), nullable=True),
        sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('training_accuracy', sa.Float(), nullable=True),
        sa.Column('validation_accuracy', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow)
    )
    
    # Create meta_learning_models table
    op.create_table(
        'meta_learning_models',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(50), default="lightgbm"),
        sa.Column('hyperparameters', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('feature_config', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('model_path', sa.String(255), nullable=True),
        sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('is_active', sa.Boolean(), default=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('user.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    )
    
    # Create rl_models table
    op.create_table(
        'rl_models',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(50), default="ppo"),
        sa.Column('hyperparameters', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('action_space', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('observation_space', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('model_path', sa.String(255), nullable=True),
        sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), default={}),
        sa.Column('is_active', sa.Boolean(), default=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('user.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    )


def downgrade():
    # Drop all tables in reverse order
    op.drop_table('rl_models')
    op.drop_table('meta_learning_models')
    op.drop_table('ml_experiment_iterations')
    op.drop_table('ml_experiments')
    op.drop_table('metric_configurations')
    op.drop_table('model_configurations')
