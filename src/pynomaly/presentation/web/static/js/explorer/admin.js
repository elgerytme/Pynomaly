document.addEventListener('DOMContentLoaded', () => {
    const adminPanel = document.querySelector('.admin-panel');
    adminPanel.innerHTML = `
        <h2>Admin Panel</h2>
        <form hx-post='/api/v1/admin/create-user' hx-target='this'>
            <label for='username'>Username:</label>
            <input type='text' name='username' required />

            <label for='role'>Role:</label>
            <select name='role'>
                <option value='user'>User</option>
                <option value='admin'>Admin</option>
            </select>

            <button type='submit'>Create User</button>
        </form>
    `;
});
