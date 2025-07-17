"""
Authentication CLI commands
"""
import click
import getpass
from typing import Optional
from ...application.use_cases.login_user_use_case import LoginUserUseCase
from ...application.dto.login_request_dto import LoginRequestDto
from ...application.services.token_service import TokenService
from ...domain.repositories.user_repository import UserRepository
from ...domain.services.authentication_service import AuthenticationService

@click.group()
def auth():
    """Authentication commands"""
    pass

@auth.command()
@click.option('--username', prompt='Username or email', help='Username or email address')
@click.option('--password', prompt=True, hide_input=True, help='Password')
@click.option('--save-token', is_flag=True, help='Save access token to file')
def login(username: str, password: str, save_token: bool):
    """Login to the system"""
    try:
        # Initialize dependencies (would be injected in real app)
        user_repo = get_user_repository()
        auth_service = AuthenticationService()
        token_service = get_token_service()
        
        login_use_case = LoginUserUseCase(user_repo, auth_service, token_service)
        
        # Execute login
        request = LoginRequestDto(identifier=username, password=password)
        response = login_use_case.execute(request)
        
        if response.success:
            click.echo("✅ Login successful!")
            click.echo(f"User ID: {response.user_id}")
            
            if save_token:
                # Save token to file
                token_file = click.get_app_dir('pynomaly') + '/token'
                with open(token_file, 'w') as f:
                    f.write(response.access_token)
                click.echo(f"Token saved to: {token_file}")
            else:
                click.echo(f"Access Token: {response.access_token}")
        else:
            click.echo(f"❌ Login failed: {response.error}")
            
    except Exception as e:
        click.echo(f"❌ Error during login: {str(e)}")

@auth.command()
def logout():
    """Logout from the system"""
    try:
        # Remove saved token file
        token_file = click.get_app_dir('pynomaly') + '/token'
        try:
            import os
            os.remove(token_file)
            click.echo("✅ Logged out successfully!")
        except FileNotFoundError:
            click.echo("ℹ️  No active session found")
            
    except Exception as e:
        click.echo(f"❌ Error during logout: {str(e)}")

@auth.command()
def whoami():
    """Show current user information"""
    try:
        # Read token from file
        token_file = click.get_app_dir('pynomaly') + '/token'
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
        except FileNotFoundError:
            click.echo("❌ Not logged in. Use 'auth login' to login.")
            return
        
        # Initialize dependencies
        token_service = get_token_service()
        user_repo = get_user_repository()
        
        # Validate token and get user info
        if not token_service.is_access_token(token):
            click.echo("❌ Invalid or expired token. Please login again.")
            return
        
        user_id = token_service.get_user_id_from_token(token)
        if not user_id:
            click.echo("❌ Invalid token. Please login again.")
            return
        
        user = user_repo.find_by_id(user_id)
        if not user:
            click.echo("❌ User not found. Please login again.")
            return
        
        # Display user information
        click.echo("👤 Current User Information:")
        click.echo(f"   ID: {user.id}")
        click.echo(f"   Username: {user.username}")
        click.echo(f"   Email: {user.email}")
        click.echo(f"   Status: {'Active' if user.is_active else 'Inactive'}")
        click.echo(f"   Verified: {'Yes' if user.is_verified else 'No'}")
        click.echo(f"   Created: {user.created_at}")
        click.echo(f"   Last Login: {user.last_login or 'Never'}")
        
    except Exception as e:
        click.echo(f"❌ Error getting user info: {str(e)}")

@auth.command()
def validate_token():
    """Validate current access token"""
    try:
        # Read token from file
        token_file = click.get_app_dir('pynomaly') + '/token'
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
        except FileNotFoundError:
            click.echo("❌ No token found. Please login first.")
            return
        
        # Initialize token service
        token_service = get_token_service()
        
        # Validate token
        if token_service.is_token_valid(token):
            payload = token_service.decode_token(token)
            click.echo("✅ Token is valid")
            click.echo(f"   Type: {payload.get('type')}")
            click.echo(f"   User ID: {payload.get('user_id')}")
            click.echo(f"   Expires: {payload.get('exp')}")
        else:
            click.echo("❌ Token is invalid or expired")
            
    except Exception as e:
        click.echo(f"❌ Error validating token: {str(e)}")

def get_user_repository() -> UserRepository:
    """Get user repository instance"""
    # This would be configured with actual implementation
    # For now, return a mock or raise NotImplementedError
    raise NotImplementedError("User repository not configured")

def get_token_service() -> TokenService:
    """Get token service instance"""
    # This would be configured with actual secret key
    # For now, use a default configuration
    return TokenService(secret_key="your-secret-key")

if __name__ == '__main__':
    auth()