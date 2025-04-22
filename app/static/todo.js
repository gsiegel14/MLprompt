
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const todoInput = document.getElementById('todo-input');
    const todoPriority = document.getElementById('todo-priority');
    const addTodoBtn = document.getElementById('add-todo-btn');
    const todoList = document.getElementById('todo-list');
    const tasksLeft = document.getElementById('tasks-left');
    const clearCompletedBtn = document.getElementById('clear-completed');
    const filterBtns = document.querySelectorAll('.filter-btn');

    // Load todos from localStorage
    let todos = JSON.parse(localStorage.getItem('mlprompt-todos') || '[]');
    let filter = 'all';

    // Render initial todos
    renderTodos();

    // Event listeners
    addTodoBtn.addEventListener('click', addTodo);
    todoInput.addEventListener('keyup', function(e) {
        if (e.key === 'Enter') addTodo();
    });
    clearCompletedBtn.addEventListener('click', clearCompleted);
    
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            filterBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            filter = this.dataset.filter;
            renderTodos();
        });
    });

    // Functions
    function addTodo() {
        const text = todoInput.value.trim();
        if (text === '') return;

        const todo = {
            id: Date.now(),
            text: text,
            priority: todoPriority.value,
            completed: false,
            createdAt: new Date().toISOString()
        };

        todos.push(todo);
        saveTodos();
        todoInput.value = '';
        renderTodos();
    }

    function toggleTodo(id) {
        todos = todos.map(todo => {
            if (todo.id === id) {
                todo.completed = !todo.completed;
            }
            return todo;
        });
        saveTodos();
        renderTodos();
    }

    function deleteTodo(id) {
        todos = todos.filter(todo => todo.id !== id);
        saveTodos();
        renderTodos();
    }

    function clearCompleted() {
        todos = todos.filter(todo => !todo.completed);
        saveTodos();
        renderTodos();
    }

    function renderTodos() {
        // Filter todos based on current filter
        let filteredTodos = todos;
        if (filter === 'active') {
            filteredTodos = todos.filter(todo => !todo.completed);
        } else if (filter === 'completed') {
            filteredTodos = todos.filter(todo => todo.completed);
        }

        // Sort by priority and creation date
        filteredTodos.sort((a, b) => {
            const priorityOrder = { high: 1, medium: 2, low: 3 };
            if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
                return priorityOrder[a.priority] - priorityOrder[b.priority];
            }
            return new Date(b.createdAt) - new Date(a.createdAt);
        });

        // Clear the list
        todoList.innerHTML = '';

        // Add todos to the list
        filteredTodos.forEach(todo => {
            const li = document.createElement('li');
            li.className = `todo-item priority-${todo.priority}`;
            if (todo.completed) li.classList.add('completed');

            li.innerHTML = `
                <input type="checkbox" ${todo.completed ? 'checked' : ''}>
                <span class="todo-text">${todo.text}</span>
                <span class="todo-priority-badge">${todo.priority}</span>
                <button class="delete-btn"><i class="fas fa-trash"></i></button>
            `;

            // Add event listeners
            li.querySelector('input[type="checkbox"]').addEventListener('change', () => toggleTodo(todo.id));
            li.querySelector('.delete-btn').addEventListener('click', () => deleteTodo(todo.id));

            todoList.appendChild(li);
        });

        // Update tasks left count
        const activeTodos = todos.filter(todo => !todo.completed);
        tasksLeft.textContent = `${activeTodos.length} task${activeTodos.length !== 1 ? 's' : ''} left`;
    }

    function saveTodos() {
        localStorage.setItem('mlprompt-todos', JSON.stringify(todos));
    }
});
