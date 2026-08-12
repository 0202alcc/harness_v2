from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Setup template directory
templates = Jinja2Templates(directory="templates")

# GET route
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

# POST route
@app.post("/send", response_class=HTMLResponse)
async def handle_form(
    request: Request, 
    field1: str = Form(...), 
    field2: str = Form(...)
):
    result_message = f"Received Field 1: '{field1}' and Field 2: '{field2}'"
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "field1": field1, 
            "field2": field2, 
            "message": result_message
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)